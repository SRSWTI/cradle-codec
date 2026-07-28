from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from cradle_codec.remote import ArtifactHttpClient, ArtifactHttpError, RemoteFetchDecodeController, serve_artifacts

from .encode import _codec_from_name

app = typer.Typer(help="Serve and fetch KVCodec artifacts over HTTP.", no_args_is_help=True)


@app.command("serve")
def serve_command(
    root: Path = typer.Option(..., "--root", "-r", exists=True, file_okay=False, dir_okay=True, help="LocalArtifactStore root to expose read-only."),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind host."),
    port: int = typer.Option(8080, "--port", min=1, help="HTTP bind port."),
) -> None:
    """Serve a LocalArtifactStore root through the artifact HTTP protocol."""

    server = serve_artifacts(root, host=host, port=port)
    bound_host, bound_port = server.server_address
    typer.echo(f"serving {root} at http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("shutdown requested", err=True)
    finally:
        server.server_close()


@app.command("fetch")
def fetch_command(
    base_url: str = typer.Option(..., "--base-url", help="Artifact HTTP base URL, e.g. http://127.0.0.1:8080."),
    source_key: str = typer.Option(..., "--source-key", help="Artifact source_key to fetch."),
    output_path: Path = typer.Option(..., "--output", "-o", file_okay=True, dir_okay=False, help="Output .npy path for the restored KV chunk."),
    codec_name: str = typer.Option("pynvvideocodec", "--codec", help="Frame codec backend: pynvvideocodec, ffmpeg, or reference."),
    variant: str | None = typer.Option(None, "--variant", help="Artifact variant to force; omit for base/adaptive selection."),
    bandwidth_bytes_per_sec: float | None = typer.Option(None, "--bandwidth-bytes-per-sec", min=1.0, help="Optional bandwidth estimate for adaptive variant selection."),
    current_variant: str | None = typer.Option(None, "--current-variant", help="Currently active variant name for switch-penalty-aware selection."),
    switch_penalty_ms: float = typer.Option(0.0, "--switch-penalty-ms", min=0.0, help="Modeled penalty for switching variants."),
    timeout_s: float = typer.Option(10.0, "--timeout-s", min=0.001, help="HTTP request timeout."),
) -> None:
    """Fetch, decode, restore, and write one remote KVCodec artifact as .npy."""

    try:
        client = ArtifactHttpClient(base_url, timeout_s=timeout_s)
        result = RemoteFetchDecodeController(client, codec=_codec_from_name(codec_name)).fetch_decode(
            source_key,
            variant_name=variant,
            bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
            current_variant_name=current_variant,
            switch_penalty_ms=switch_penalty_ms,
        )
    except (ArtifactHttpError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.kv)
    typer.echo(
        " ".join(
            (
                f"wrote {output_path}",
                f"shape={tuple(result.kv.shape)}",
                f"dtype={result.kv.dtype}",
                f"variant={result.variant_name or 'base'}",
                f"transfer_ms={result.timing.transfer_ms:.3f}",
                f"decode_ms={result.timing.decode_ms:.3f}",
                f"restore_ms={result.timing.restore_ms:.3f}",
            )
        )
    )


if __name__ == "__main__":
    app()
