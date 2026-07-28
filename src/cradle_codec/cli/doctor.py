from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import typer

from cradle_codec import __version__
from cradle_codec.gstreamer import gstreamer_available


app = typer.Typer(
    help="Inspect local Cradle Codec production-readiness prerequisites.",
    no_args_is_help=False,
)


@dataclass(frozen=True, slots=True)
class CommandStatus:
    available: bool
    path: str | None = None
    returncode: int | None = None
    stdout_first_line: str | None = None
    stderr_first_line: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    available: bool
    origin: str | None = None


def _module_status(name: str) -> ModuleStatus:
    spec = importlib.util.find_spec(name)
    return ModuleStatus(
        available=spec is not None, origin=None if spec is None else spec.origin
    )


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _command_status(
    executable: str, args: Sequence[str], *, timeout_s: float
) -> CommandStatus:
    path = shutil.which(executable)
    if path is None:
        return CommandStatus(available=False)
    try:
        result = subprocess.run(
            [path, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - platform/runtime dependent
        return CommandStatus(
            available=True, path=path, error=f"{type(exc).__name__}: {exc}"
        )
    return CommandStatus(
        available=True,
        path=path,
        returncode=result.returncode,
        stdout_first_line=_first_line(result.stdout),
        stderr_first_line=_first_line(result.stderr),
    )


def build_doctor_report(*, timeout_s: float = 5.0) -> dict[str, Any]:
    """Return a JSON-serializable production-readiness report."""

    modules = {
        name: asdict(_module_status(name))
        for name in (
            "cradle_codec",
            "numpy",
            "typer",
            "gi",
            "PyNvVideoCodec",
            "torch",
            "transformers",
            "accelerate",
            "vllm",
            "lmcache",
        )
    }
    commands = {
        "cradle-codec": asdict(
            _command_status("cradle-codec", ["--help"], timeout_s=timeout_s)
        ),
        "vllm": asdict(_command_status("vllm", ["--version"], timeout_s=timeout_s)),
        "lmcache": asdict(_command_status("lmcache", ["--help"], timeout_s=timeout_s)),
        "nvidia-smi": asdict(_command_status("nvidia-smi", [], timeout_s=timeout_s)),
        "gst-launch-1.0": asdict(
            _command_status("gst-launch-1.0", ["--version"], timeout_s=timeout_s)
        ),
    }
    ready = all(
        modules[name]["available"] for name in ("cradle_codec", "numpy", "typer")
    )
    serving_ready = all(
        modules[name]["available"]
        for name in ("torch", "transformers", "vllm", "lmcache")
    )
    nvenc_ready = (
        modules["PyNvVideoCodec"]["available"] and commands["nvidia-smi"]["available"]
    )
    gstreamer_ready = gstreamer_available() and commands["gst-launch-1.0"]["available"]
    return {
        "package": "cradle-codec",
        "version": __version__,
        "python": sys.version.split()[0],
        "core_artifact_tools_ready": ready,
        "serving_optional_dependencies_ready": serving_ready,
        "gpu_video_codec_path_detected": nvenc_ready,
        "gstreamer_packet_transport_ready": gstreamer_ready,
        "modules": modules,
        "commands": commands,
    }


@app.callback(invoke_without_command=True)
def doctor_command(
    timeout_s: float = typer.Option(
        5.0, "--timeout-s", min=0.1, help="Per-command probe timeout in seconds."
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--compact", help="Pretty-print JSON output."
    ),
) -> None:
    """Print a local dependency and command probe report."""

    report = build_doctor_report(timeout_s=timeout_s)
    typer.echo(json.dumps(report, sort_keys=True, indent=2 if pretty else None))
