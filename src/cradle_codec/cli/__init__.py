from __future__ import annotations

import typer

from .bench import app as bench_app
from .decode import app as decode_app
from .doctor import app as doctor_app
from .gstreamer import app as gstreamer_app
from .integration import app as integration_app
from .encode import app as encode_app
from .remote import app as remote_app
from .live import app as live_app

app = typer.Typer(help="KVCodec artifact tools", no_args_is_help=True)
app.add_typer(
    encode_app, name="encode", help="Encode a KV .npy chunk into an artifact directory."
)
app.add_typer(bench_app, name="bench", help="Estimate reuse TTFT from an artifact.")
app.add_typer(
    integration_app, name="integration", help="Print vLLM/LMCache integration configs."
)
app.add_typer(
    doctor_app, name="doctor", help="Inspect local production-readiness prerequisites."
)
app.add_typer(
    gstreamer_app, name="gstreamer", help="Run the optional GStreamer packet transport."
)
app.add_typer(
    decode_app, name="decode", help="Decode an artifact directory into a KV .npy chunk."
)
app.add_typer(live_app, name="live", help="Run real-model TTFT/TPOT benchmarks.")
app.add_typer(
    remote_app, name="remote", help="Serve and fetch KVCodec artifacts over HTTP."
)


@app.callback()
def main() -> None:
    """KVCodec artifact tools."""


@app.command("version")
def version_command() -> None:
    """Print the package version."""
    from cradle_codec import __version__

    typer.echo(__version__)
