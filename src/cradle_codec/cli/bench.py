from __future__ import annotations

import typer
from pathlib import Path

import numpy as np

from cradle_codec.benchmark import benchmark_artifact_reuse

app = typer.Typer(help="Estimate KVCodec reuse TTFT from a local artifact.", no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(
    artifact_dir: Path = typer.Option(..., "--artifact", "-a", exists=True, file_okay=False, dir_okay=True, help="KVCodec artifact directory."),
    expected_path: Path | None = typer.Option(None, "--expected", "-e", exists=True, file_okay=True, dir_okay=False, help="Optional original KV .npy chunk for accuracy measurement."),
    bandwidth_bytes_per_sec: float = typer.Option(..., "--bandwidth-bytes-per-sec", min=1.0, help="Simulated network bandwidth."),
    prefill_ms: float = typer.Option(..., "--prefill-ms", min=0.000001, help="Observed or modeled full-prefill TTFT."),
    scheduler_wait_ms: float = typer.Option(0.0, "--scheduler-wait-ms", min=0.0, help="Optional modeled scheduler wait before transfer."),
    variant: str | None = typer.Option(None, "--variant", help="Artifact variant to force; omit for adaptive selection."),
) -> None:
    """Print a JSON report for full prefill, raw KV reuse, and codec reuse."""

    expected = np.load(expected_path) if expected_path is not None else None
    report = benchmark_artifact_reuse(
        artifact_dir,
        expected_kv=expected,
        bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
        prefill_ms=prefill_ms,
        scheduler_wait_ms=scheduler_wait_ms,
        variant_name=variant,
    )
    typer.echo(report.to_json())
