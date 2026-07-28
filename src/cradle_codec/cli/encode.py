from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from cradle_codec.codec import FFmpegHEVCCodec, PyNvVideoCodecHEVCCodec, RawReferenceCodec
from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.pipeline import encode_kv_chunk
from cradle_codec.quant import QuantizationSpec

app = typer.Typer(help="Encode a canonical [2,L,T,H,D] KV .npy chunk into a KVCodec artifact.")


def _codec_from_name(name: str, *, nvenc_workers: int = 1, nvdec_workers: int = 1):
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"pynv", "pynvvideocodec", "pynv-hevc", "nvenc"}:
        return PyNvVideoCodecHEVCCodec(nvenc_workers=nvenc_workers, nvdec_workers=nvdec_workers)
    if normalized in {"ffmpeg", "ffmpeg-hevc", "libx265"}:
        return FFmpegHEVCCodec()
    if normalized in {"reference", "raw-reference", "raw"}:
        return RawReferenceCodec()
    raise typer.BadParameter("codec must be one of: pynvvideocodec, ffmpeg, reference")


@app.callback(invoke_without_command=True)
def main(
    input_path: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False, help="Input .npy KV chunk with shape [2,L,T,H,D]."),
    output_dir: Path = typer.Option(..., "--output", "-o", file_okay=False, dir_okay=True, help="Artifact directory to create."),
    source_key: str = typer.Option("local-chunk", help="Stable source key stored in the manifest."),
    model: str = typer.Option("unknown", help="Model name stored in the manifest."),
    layers_per_frame: int = typer.Option(3, min=1, max=3, help="Number of adjacent layers mapped to frame channels."),
    head_rows: int = typer.Option(..., min=1, help="Attention-head tiling rows."),
    head_cols: int = typer.Option(..., min=1, help="Attention-head tiling columns."),
    dim_rows: int = typer.Option(..., min=1, help="Head-dimension tiling rows."),
    dim_cols: int = typer.Option(..., min=1, help="Head-dimension tiling columns."),
    quant_mode: str = typer.Option("uint8_minmax", help="Quantization mode. Currently supported: uint8_minmax."),
    quant_axis: str = typer.Option("channel", help="uint8_minmax metadata axis: part, frame, or channel."),
    codec_name: str = typer.Option(
        "pynvvideocodec",
        "--codec",
        help="Frame codec backend: pynvvideocodec (NVENC HEVC), ffmpeg, or reference.",
    ),
    nvenc_workers: int = typer.Option(1, "--nvenc-workers", min=1, help="PyNvVideoCodec NVENC worker count."),
    nvdec_workers: int = typer.Option(1, "--nvdec-workers", min=1, help="PyNvVideoCodec NVDEC worker count recorded for artifact decode."),
) -> None:
    kv = np.load(input_path)
    layout = KVCodecLayout(
        num_layers=int(kv.shape[1]),
        num_kv_heads=int(kv.shape[3]),
        head_dim=int(kv.shape[4]),
        layers_per_frame=layers_per_frame,
        tiling=HeadDimTiling(head_rows=head_rows, head_cols=head_cols, dim_rows=dim_rows, dim_cols=dim_cols),
    )
    manifest = encode_kv_chunk(
        kv,
        output_dir,
        source_key=source_key,
        model=model,
        layout=layout,
        quantization=QuantizationSpec(mode=quant_mode, axis=quant_axis),  # type: ignore[arg-type]
        codec=_codec_from_name(codec_name, nvenc_workers=nvenc_workers, nvdec_workers=nvdec_workers),
    )
    total_bytes = sum(part.payload_bytes for part in manifest.parts)
    typer.echo(f"wrote {output_dir / 'manifest.json'}")
    typer.echo(f"parts={len(manifest.parts)} payload_bytes={total_bytes}")


if __name__ == "__main__":
    app()
