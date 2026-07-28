from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from cradle_codec.codec import EncodedPayload, FFmpegHEVCCodec, FrameCodec, PyNvVideoCodecHEVCCodec, RawReferenceCodec
from cradle_codec.layout import (
    FrameBatch,
    FrameGeometry,
    HeadDimTiling,
    KVCodecLayout,
    LayerGroup,
    PackedFrameBatch,
    unpack_frame_batches_to_kv,
)
from cradle_codec.manifest import (
    ArtifactManifest,
    ArtifactPart,
    CodecManifest,
    LayoutManifest,
    PartQuantizationManifest,
    read_manifest,
    verify_part_payload,
)
from cradle_codec.quant import QuantizationMetadata, dequantize_uint8_minmax

PartKey = tuple[str, int]

def _layout_manifest_for(manifest: ArtifactManifest, variant_name: str | None = None) -> LayoutManifest:
    if variant_name is None:
        return manifest.layout
    return manifest.variant(variant_name).layout or manifest.layout


def layout_from_manifest(manifest: ArtifactManifest, *, variant_name: str | None = None) -> KVCodecLayout:
    layout = _layout_manifest_for(manifest, variant_name=variant_name)
    return KVCodecLayout(
        num_layers=manifest.kv_shape.num_layers,
        num_kv_heads=manifest.kv_shape.num_kv_heads,
        head_dim=manifest.kv_shape.head_dim,
        layers_per_frame=layout.layers_per_frame,
        tiling=HeadDimTiling(
            head_rows=layout.head_rows,
            head_cols=layout.head_cols,
            dim_rows=layout.dim_rows,
            dim_cols=layout.dim_cols,
        ),
        token_axis_is_time=layout.token_axis_is_time,
    )


def _metadata_for_part(part: ArtifactPart) -> FrameBatch:
    side = {"k": 0, "v": 1}.get(part.side)
    if side is None:
        raise ValueError(f"invalid part side {part.side!r}")
    return FrameBatch(
        side=side,
        side_name=part.side,  # type: ignore[arg-type]
        layer_group=LayerGroup(group_index=part.layer_group_index, layer_indices=part.layer_indices),
        token_start=part.token_start,
        token_count=part.token_count,
        geometry=FrameGeometry(
            logical_height=part.logical_height,
            logical_width=part.logical_width,
            encoded_height=part.encoded_height,
            encoded_width=part.encoded_width,
        ),
    )


def _quant_metadata(part_quant: PartQuantizationManifest) -> QuantizationMetadata:
    min_values = None if part_quant.min_values is None else np.asarray(part_quant.min_values, dtype=np.float32)
    scales = None if part_quant.scales is None else np.asarray(part_quant.scales, dtype=np.float32)
    return QuantizationMetadata(
        mode=part_quant.mode,
        axis=part_quant.axis,
        min_values=min_values,
        scales=scales,
        source_dtype=part_quant.source_dtype,
        transport_dtype=part_quant.transport_dtype,
    )


def _dequantize_frames(frames: np.ndarray, part_quant: PartQuantizationManifest) -> np.ndarray:
    if part_quant.mode != "uint8_minmax":
        raise ValueError(f"unsupported quantization mode: {part_quant.mode}")
    return dequantize_uint8_minmax(frames, _quant_metadata(part_quant))


def _codec_manifest_for(manifest: ArtifactManifest, variant_name: str | None = None) -> CodecManifest:
    if variant_name is None:
        return manifest.codec
    return manifest.variant(variant_name).codec or manifest.codec


def _codec_for_manifest(manifest: ArtifactManifest, *, variant_name: str | None = None) -> FrameCodec:
    codec_manifest = _codec_manifest_for(manifest, variant_name=variant_name)
    backend = codec_manifest.backend
    if codec_manifest.family == "reference" or backend == RawReferenceCodec.codec_name:
        return RawReferenceCodec()
    if backend == FFmpegHEVCCodec.codec_name:
        return FFmpegHEVCCodec()
    if backend == PyNvVideoCodecHEVCCodec.codec_name:
        params = codec_manifest.codec_params
        return PyNvVideoCodecHEVCCodec(
            nvenc_workers=int(params.get("nvenc_workers", 1)),
            nvdec_workers=int(params.get("nvdec_workers", 1)),
        )
    raise ValueError(f"unsupported artifact codec backend {backend!r}")


def part_key(part: ArtifactPart) -> PartKey:
    return (part.side, part.layer_group_index)


def select_manifest_parts(
    manifest: ArtifactManifest,
    *,
    variant_name: str | None = None,
    part_keys: Iterable[PartKey] | None = None,
) -> tuple[ArtifactPart, ...]:
    selected_parts = manifest.sorted_parts(variant_name=variant_name)
    if part_keys is None:
        return selected_parts
    requested = set(part_keys)
    parts = tuple(part for part in selected_parts if part_key(part) in requested)
    found = {part_key(part) for part in parts}
    missing = requested - found
    if missing:
        raise KeyError(f"requested artifact parts are not present in variant {variant_name or 'base'}: {sorted(missing)}")
    return parts


def validate_full_manifest_parts(manifest: ArtifactManifest, *, variant_name: str | None = None) -> tuple[ArtifactPart, ...]:
    """Validate that a full-artifact restore has exactly the expected side/group parts."""

    parts = select_manifest_parts(manifest, variant_name=variant_name)
    layout = layout_from_manifest(manifest, variant_name=variant_name)
    expected_groups = {group.group_index: group.layer_indices for group in layout.layer_groups()}
    expected_keys = {(side, group_index) for side in ("k", "v") for group_index in expected_groups}
    seen: dict[PartKey, ArtifactPart] = {}
    for part in parts:
        key = part_key(part)
        if key in seen:
            raise ValueError(f"duplicate artifact part {key!r} in variant {variant_name or 'base'}")
        seen[key] = part
        expected_layers = expected_groups.get(part.layer_group_index)
        if expected_layers is None:
            raise ValueError(
                f"unexpected layer_group_index {part.layer_group_index} in variant {variant_name or 'base'}"
            )
        if tuple(part.layer_indices) != expected_layers:
            raise ValueError(
                f"layer indices for part {key!r} do not match layout: "
                f"{tuple(part.layer_indices)} != {expected_layers}"
            )
        if part.token_start != 0 or part.token_count != manifest.kv_shape.num_tokens:
            raise ValueError(
                f"part {key!r} token span {part.token_start}:{part.token_start + part.token_count} "
                f"does not cover full chunk length {manifest.kv_shape.num_tokens}"
            )

    actual_keys = set(seen)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        raise ValueError(
            f"artifact variant {variant_name or 'base'} does not match layout parts: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return parts


def decode_part_payload(
    manifest: ArtifactManifest,
    part: ArtifactPart,
    data: bytes,
    *,
    codec: FrameCodec | None = None,
    variant_name: str | None = None,
) -> PackedFrameBatch:
    codec_manifest = _codec_manifest_for(manifest, variant_name=variant_name)
    codec = codec or _codec_for_manifest(manifest, variant_name=variant_name)
    payload = EncodedPayload(
        codec_name=codec_manifest.backend,
        data=data,
        payload_bytes=part.payload_bytes,
        checksum=part.checksum,
    )
    metadata = _metadata_for_part(part)
    decoded_frames = codec.decode(payload, metadata.geometry)
    restored_frames = _dequantize_frames(decoded_frames, part.quantization)
    return PackedFrameBatch(metadata=metadata, frames=restored_frames)


def iter_decoded_artifact_parts(
    artifact_dir: str | Path,
    *,
    manifest: ArtifactManifest | None = None,
    codec: FrameCodec | None = None,
    variant_name: str | None = None,
    part_keys: Iterable[PartKey] | None = None,
) -> Iterable[PackedFrameBatch]:
    artifact_dir = Path(artifact_dir)
    manifest = manifest or read_manifest(artifact_dir / "manifest.json")
    codec = codec or _codec_for_manifest(manifest, variant_name=variant_name)
    for part in select_manifest_parts(manifest, variant_name=variant_name, part_keys=part_keys):
        data = verify_part_payload(artifact_dir, part)
        yield decode_part_payload(manifest, part, data, codec=codec, variant_name=variant_name)

def decode_artifact_parts(
    artifact_dir: str | Path,
    *,
    manifest: ArtifactManifest | None = None,
    codec: FrameCodec | None = None,
    variant_name: str | None = None,
    part_keys: Iterable[PartKey] | None = None,
) -> list[PackedFrameBatch]:
    artifact_dir = Path(artifact_dir)
    manifest = manifest or read_manifest(artifact_dir / "manifest.json")
    codec_manifest = _codec_manifest_for(manifest, variant_name=variant_name)
    codec = codec or _codec_for_manifest(manifest, variant_name=variant_name)
    parts = select_manifest_parts(manifest, variant_name=variant_name, part_keys=part_keys)

    decode_many = getattr(codec, "decode_many", None)
    if not callable(decode_many):
        return [
            decode_part_payload(
                manifest,
                part,
                verify_part_payload(artifact_dir, part),
                codec=codec,
                variant_name=variant_name,
            )
            for part in parts
        ]

    payload_requests: list[tuple[EncodedPayload, FrameGeometry]] = []
    metadata_by_part: list[FrameBatch] = []
    for part in parts:
        data = verify_part_payload(artifact_dir, part)
        metadata = _metadata_for_part(part)
        payload_requests.append(
            (
                EncodedPayload(
                    codec_name=codec_manifest.backend,
                    data=data,
                    payload_bytes=part.payload_bytes,
                    checksum=part.checksum,
                ),
                metadata.geometry,
            )
        )
        metadata_by_part.append(metadata)

    decoded_batches = decode_many(payload_requests)
    return [
        PackedFrameBatch(metadata=metadata, frames=_dequantize_frames(decoded_frames, part.quantization))
        for part, metadata, decoded_frames in zip(parts, metadata_by_part, decoded_batches, strict=True)
    ]


def decode_kv_artifact(artifact_dir: str | Path, *, codec: FrameCodec | None = None, variant_name: str | None = None) -> np.ndarray:
    artifact_dir = Path(artifact_dir)
    manifest = read_manifest(artifact_dir / "manifest.json")
    validate_full_manifest_parts(manifest, variant_name=variant_name)
    layout = layout_from_manifest(manifest, variant_name=variant_name)
    batches = decode_artifact_parts(artifact_dir, manifest=manifest, codec=codec, variant_name=variant_name)
    return unpack_frame_batches_to_kv(batches, layout, num_tokens=manifest.kv_shape.num_tokens)
