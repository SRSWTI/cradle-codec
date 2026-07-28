from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from cradle_codec.pipeline import decode_kv_artifact

app = typer.Typer(help="Decode a KVCodec artifact directory back to a canonical [2,L,T,H,D] .npy chunk.")


@app.callback(invoke_without_command=True)
def main(
    artifact_dir: Path = typer.Option(..., "--artifact", "-a", exists=True, file_okay=False, dir_okay=True, help="Artifact directory containing manifest.json."),
    output_path: Path = typer.Option(..., "--output", "-o", file_okay=True, dir_okay=False, help="Output .npy file."),
) -> None:
    kv = decode_kv_artifact(artifact_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, kv)
    typer.echo(f"wrote {output_path} shape={tuple(kv.shape)} dtype={kv.dtype}")


if __name__ == "__main__":
    app()
