from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cradle_codec.codec import FFmpegHEVCCodec, FrameCodec, PyNvVideoCodecHEVCCodec, RawReferenceCodec
from cradle_codec.layout import KVCodecLayout, KVShape, PackedFrameBatch, pack_kv_to_frame_batches
from cradle_codec.manifest import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactPart,
    ArtifactVariant,
    CodecManifest,
    KVShapeManifest,
    LayoutManifest,
    PartQuantizationManifest,
    QuantizationManifest,
    write_manifest,
)
from cradle_codec.quant import QuantizationMetadata, QuantizationSpec, quantize_uint8_minmax

_SAFE_VARIANT_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass(frozen=True)
class VariantEncodeSpec:
    """One independently decodable artifact variant for the same logical KV chunk."""

    name: str
    layout: KVCodecLayout
    quantization: QuantizationSpec | None = None
    codec: FrameCodec | None = None
    estimated_decode_ms: float | None = None


def _layout_manifest(layout: KVCodecLayout) -> LayoutManifest:
    return LayoutManifest(
        layers_per_frame=layout.layers_per_frame,
        head_rows=layout.tiling.head_rows,
        head_cols=layout.tiling.head_cols,
        dim_rows=layout.tiling.dim_rows,
        dim_cols=layout.tiling.dim_cols,
        token_axis_is_time=layout.token_axis_is_time,
    )


def _kv_shape_manifest(kv: np.ndarray) -> KVShapeManifest:
    if kv.ndim != 5:
        raise ValueError(f"expected KV rank 5 [2,L,T,H,D], got rank {kv.ndim}")
    shape = KVShape(kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3], kv.shape[4])
    return KVShapeManifest(*shape.as_tuple)


def _part_quant_manifest(metadata: QuantizationMetadata) -> PartQuantizationManifest:
    min_values = None if metadata.min_values is None else metadata.min_values
    scales = None if metadata.scales is None else metadata.scales
    return PartQuantizationManifest(
        mode=metadata.mode,
        axis=metadata.axis,
        min_values=min_values,
        scales=scales,
        source_dtype=metadata.source_dtype,
        transport_dtype=metadata.transport_dtype,
    )


def _quant_manifest(spec: QuantizationSpec) -> QuantizationManifest:
    return QuantizationManifest(mode=spec.mode, axis=spec.axis)


def _quantize_frames(frames: np.ndarray, spec: QuantizationSpec) -> tuple[np.ndarray, PartQuantizationManifest]:
    if spec.mode != "uint8_minmax":
        raise ValueError(f"unsupported quantization mode: {spec.mode}")
    encoded, metadata = quantize_uint8_minmax(frames, axis=spec.axis)
    return encoded, _part_quant_manifest(metadata)


def _codec_manifest(codec: FrameCodec) -> CodecManifest:
    if isinstance(codec, RawReferenceCodec):
        return CodecManifest(family="reference", backend=codec.codec_name, lossless_video=True, codec_params={})
    if isinstance(codec, FFmpegHEVCCodec):
        return CodecManifest(
            family="hevc",
            backend=codec.codec_name,
            lossless_video=True,
            codec_params={"encoder": codec.encoder, "preset": codec.preset},
        )
    if isinstance(codec, PyNvVideoCodecHEVCCodec):
        return CodecManifest(
            family="hevc",
            backend=codec.codec_name,
            lossless_video=True,
            codec_params={
                "encoder": "pynvvideocodec",
                "preset": codec.preset,
                "tuning_info": codec.tuning_info,
                "rc": codec.rc,
                "constqp": codec.constqp,
                "gop": codec.gop,
                "bf": codec.bf,
                "nvenc_workers": codec.nvenc_workers,
                "nvdec_workers": codec.nvdec_workers,
            },
        )
    return CodecManifest(family="custom", backend=codec.codec_name, lossless_video=True, codec_params={})


def _payload_geometry(codec: FrameCodec, geometry):
    adjust = getattr(codec, "payload_geometry", None)
    if callable(adjust):
        return adjust(geometry)
    return geometry


def _variant_dir_name(name: str) -> str:
    cleaned = _SAFE_VARIANT_CHARS.sub("-", name.strip()).strip("-._=")
    if not cleaned:
        raise ValueError("variant name must contain at least one safe path character")
    return cleaned[:96]


def _encode_parts(
    kv: np.ndarray,
    artifact_dir: Path,
    *,
    layout: KVCodecLayout,
    quantization: QuantizationSpec,
    codec: FrameCodec,
    relative_parts_dir: Path,
) -> tuple[ArtifactPart, ...]:
    parts_dir = artifact_dir / relative_parts_dir
    parts_dir.mkdir(parents=True, exist_ok=True)

    prepared: list[tuple[PackedFrameBatch, np.ndarray, object, object, str]] = []
    for batch in pack_kv_to_frame_batches(kv, layout):
        quantized_frames, quant_manifest = _quantize_frames(batch.frames, quantization)
        payload_geometry = _payload_geometry(codec, batch.metadata.geometry)
        file_name = f"{batch.metadata.side_name}.g{batch.metadata.layer_group.group_index}.bin"
        prepared.append((batch, quantized_frames, quant_manifest, payload_geometry, file_name))

    encode_many = getattr(codec, "encode_many", None)
    if callable(encode_many):
        payloads = encode_many((frames, geometry) for _, frames, _, geometry, _ in prepared)
    else:
        payloads = tuple(codec.encode(frames, geometry) for _, frames, _, geometry, _ in prepared)

    parts: list[ArtifactPart] = []
    for (batch, _quantized_frames, quant_manifest, payload_geometry, file_name), payload in zip(prepared, payloads, strict=True):
        payload_path = parts_dir / file_name
        payload_path.write_bytes(payload.data)
        parts.append(
            ArtifactPart(
                side=batch.metadata.side_name,
                layer_group_index=batch.metadata.layer_group.group_index,
                layer_indices=batch.metadata.layer_group.layer_indices,
                token_start=batch.metadata.token_start,
                token_count=batch.metadata.token_count,
                logical_height=payload_geometry.logical_height,
                logical_width=payload_geometry.logical_width,
                encoded_height=payload_geometry.encoded_height,
                encoded_width=payload_geometry.encoded_width,
                payload_path=relative_parts_dir.joinpath(file_name).as_posix(),
                payload_bytes=payload.payload_bytes,
                checksum=payload.checksum,
                quantization=quant_manifest,
            )
        )
    return tuple(parts)


def encode_kv_chunk(
    kv: np.ndarray,
    artifact_dir: str | Path,
    *,
    source_key: str,
    model: str,
    layout: KVCodecLayout,
    quantization: QuantizationSpec | None = None,
    codec: FrameCodec | None = None,
) -> ArtifactManifest:
    """Encode one canonical KV chunk into a self-describing local artifact.

    The default path mirrors the paper's practical pipeline: quantize KV values
    to integer video frames, then rely on a lossless frame/video codec stage.
    """

    artifact_dir = Path(artifact_dir)
    quantization = quantization or QuantizationSpec(mode="uint8_minmax", axis="channel")
    codec = codec or RawReferenceCodec()
    parts = _encode_parts(
        kv,
        artifact_dir,
        layout=layout,
        quantization=quantization,
        codec=codec,
        relative_parts_dir=Path("parts"),
    )
    manifest = ArtifactManifest(
        version=SCHEMA_VERSION,
        source_key=source_key,
        model=model,
        kv_shape=_kv_shape_manifest(kv),
        layout=_layout_manifest(layout),
        quantization=_quant_manifest(quantization),
        codec=_codec_manifest(codec),
        parts=parts,
        variants=(ArtifactVariant(name="base", parts=parts, payload_bytes=sum(part.payload_bytes for part in parts)),),
    )
    write_manifest(artifact_dir / "manifest.json", manifest)
    return manifest


def encode_kv_chunk_variants(
    kv: np.ndarray,
    artifact_dir: str | Path,
    *,
    source_key: str,
    model: str,
    variants: Iterable[VariantEncodeSpec],
) -> ArtifactManifest:
    """Encode one KV chunk into multiple independently decodable variants.

    The first variant is also mirrored into the top-level manifest fields for
    backwards-compatible readers. Every variant may carry its own layout,
    quantization policy, codec backend, and decode-time estimate.
    """

    artifact_dir = Path(artifact_dir)
    specs = tuple(variants)
    if not specs:
        raise ValueError("at least one artifact variant is required")

    encoded_variants: list[ArtifactVariant] = []
    for spec in specs:
        quantization = spec.quantization or QuantizationSpec(mode="uint8_minmax", axis="channel")
        codec = spec.codec or RawReferenceCodec()
        parts = _encode_parts(
            kv,
            artifact_dir,
            layout=spec.layout,
            quantization=quantization,
            codec=codec,
            relative_parts_dir=Path("variants") / _variant_dir_name(spec.name),
        )
        encoded_variants.append(
            ArtifactVariant(
                name=spec.name,
                parts=parts,
                payload_bytes=sum(part.payload_bytes for part in parts),
                estimated_decode_ms=spec.estimated_decode_ms,
                layout=_layout_manifest(spec.layout),
                quantization=_quant_manifest(quantization),
                codec=_codec_manifest(codec),
            )
        )

    primary = encoded_variants[0]
    manifest = ArtifactManifest(
        version=SCHEMA_VERSION,
        source_key=source_key,
        model=model,
        kv_shape=_kv_shape_manifest(kv),
        layout=primary.layout or _layout_manifest(specs[0].layout),
        quantization=primary.quantization or _quant_manifest(specs[0].quantization or QuantizationSpec(mode="uint8_minmax", axis="channel")),
        codec=primary.codec or _codec_manifest(specs[0].codec or RawReferenceCodec()),
        parts=primary.parts,
        variants=tuple(encoded_variants),
    )
    write_manifest(artifact_dir / "manifest.json", manifest)
    return manifest
