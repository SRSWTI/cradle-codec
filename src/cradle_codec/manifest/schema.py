from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KVShapeManifest:
    num_sides: int
    num_layers: int
    num_tokens: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class LayoutManifest:
    layers_per_frame: int
    head_rows: int
    head_cols: int
    dim_rows: int
    dim_cols: int
    token_axis_is_time: bool = True


@dataclass(frozen=True)
class QuantizationManifest:
    mode: str
    axis: str


@dataclass(frozen=True)
class PartQuantizationManifest:
    mode: str
    axis: str
    min_values: Any | None
    scales: Any | None
    source_dtype: str
    transport_dtype: str


@dataclass(frozen=True)
class CodecManifest:
    family: str
    backend: str
    lossless_video: bool
    codec_params: dict[str, Any]


@dataclass(frozen=True)
class ArtifactPart:
    side: str
    layer_group_index: int
    layer_indices: tuple[int, ...]
    token_start: int
    token_count: int
    logical_height: int
    logical_width: int
    encoded_height: int
    encoded_width: int
    payload_path: str
    payload_bytes: int
    checksum: str
    quantization: PartQuantizationManifest

    @property
    def ordering_key(self) -> tuple[int, int]:
        side_order = {"k": 0, "v": 1}
        return (side_order[self.side], self.layer_group_index)


@dataclass(frozen=True)
class ArtifactVariant:
    name: str
    parts: tuple[ArtifactPart, ...]
    payload_bytes: int
    estimated_decode_ms: float | None = None
    layout: LayoutManifest | None = None
    quantization: QuantizationManifest | None = None
    codec: CodecManifest | None = None

    def sorted_parts(self) -> tuple[ArtifactPart, ...]:
        return tuple(sorted(self.parts, key=lambda part: part.ordering_key))


@dataclass(frozen=True)
class ArtifactManifest:
    version: int
    source_key: str
    model: str
    kv_shape: KVShapeManifest
    layout: LayoutManifest
    quantization: QuantizationManifest
    codec: CodecManifest
    parts: tuple[ArtifactPart, ...]
    variants: tuple[ArtifactVariant, ...] = ()

    def sorted_parts(self, variant_name: str | None = None) -> tuple[ArtifactPart, ...]:
        if variant_name is None:
            return tuple(sorted(self.parts, key=lambda part: part.ordering_key))
        return self.variant(variant_name).sorted_parts()

    def base_variant(self, name: str = "base") -> ArtifactVariant:
        return ArtifactVariant(
            name=name,
            parts=self.sorted_parts(),
            payload_bytes=sum(part.payload_bytes for part in self.parts),
            estimated_decode_ms=None,
        )

    def variant(self, name: str) -> ArtifactVariant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        if name == "base":
            return self.base_variant()
        raise KeyError(f"unknown artifact variant {name!r}")

    def variants_with_base(self) -> tuple[ArtifactVariant, ...]:
        base = self.base_variant()
        variants = [base]
        for variant in self.variants:
            if variant.name == base.name:
                variants[0] = variant
            else:
                variants.append(variant)
        return tuple(variants)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _part_to_dict(part: ArtifactPart) -> dict[str, Any]:
    return {
        "side": part.side,
        "layer_group_index": part.layer_group_index,
        "layer_indices": _jsonable(part.layer_indices),
        "token_start": part.token_start,
        "token_count": part.token_count,
        "logical_height": part.logical_height,
        "logical_width": part.logical_width,
        "encoded_height": part.encoded_height,
        "encoded_width": part.encoded_width,
        "payload_path": part.payload_path,
        "payload_bytes": part.payload_bytes,
        "checksum": part.checksum,
        "quantization": _jsonable(
            {
                "mode": part.quantization.mode,
                "axis": part.quantization.axis,
                "min_values": part.quantization.min_values,
                "scales": part.quantization.scales,
                "source_dtype": part.quantization.source_dtype,
                "transport_dtype": part.quantization.transport_dtype,
            }
        ),
    }


def _variant_to_dict(variant: ArtifactVariant) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": variant.name,
        "parts": [_part_to_dict(part) for part in variant.sorted_parts()],
        "payload_bytes": variant.payload_bytes,
    }
    if variant.estimated_decode_ms is not None:
        data["estimated_decode_ms"] = variant.estimated_decode_ms
    if variant.layout is not None:
        data["layout"] = {
            "layers_per_frame": variant.layout.layers_per_frame,
            "head_rows": variant.layout.head_rows,
            "head_cols": variant.layout.head_cols,
            "dim_rows": variant.layout.dim_rows,
            "dim_cols": variant.layout.dim_cols,
            "token_axis_is_time": variant.layout.token_axis_is_time,
        }
    if variant.quantization is not None:
        data["quantization"] = {"mode": variant.quantization.mode, "axis": variant.quantization.axis}
    if variant.codec is not None:
        data["codec"] = {
            "family": variant.codec.family,
            "backend": variant.codec.backend,
            "lossless_video": variant.codec.lossless_video,
            "codec_params": variant.codec.codec_params,
        }
    return _jsonable(data)


def manifest_to_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": manifest.version,
        "source_key": manifest.source_key,
        "model": manifest.model,
        "kv_shape": _jsonable(
            {
                "num_sides": manifest.kv_shape.num_sides,
                "num_layers": manifest.kv_shape.num_layers,
                "num_tokens": manifest.kv_shape.num_tokens,
                "num_kv_heads": manifest.kv_shape.num_kv_heads,
                "head_dim": manifest.kv_shape.head_dim,
            }
        ),
        "layout": _jsonable(
            {
                "layers_per_frame": manifest.layout.layers_per_frame,
                "head_rows": manifest.layout.head_rows,
                "head_cols": manifest.layout.head_cols,
                "dim_rows": manifest.layout.dim_rows,
                "dim_cols": manifest.layout.dim_cols,
                "token_axis_is_time": manifest.layout.token_axis_is_time,
            }
        ),
        "quantization": {"mode": manifest.quantization.mode, "axis": manifest.quantization.axis},
        "codec": _jsonable(
            {
                "family": manifest.codec.family,
                "backend": manifest.codec.backend,
                "lossless_video": manifest.codec.lossless_video,
                "codec_params": manifest.codec.codec_params,
            }
        ),
        "parts": [_part_to_dict(part) for part in manifest.sorted_parts()],
    }
    if manifest.variants:
        data["variants"] = [_variant_to_dict(variant) for variant in manifest.variants]
    return _jsonable(data)


def _part_quant_from_dict(data: dict[str, Any]) -> PartQuantizationManifest:
    return PartQuantizationManifest(
        mode=str(data["mode"]),
        axis=str(data["axis"]),
        min_values=data.get("min_values"),
        scales=data.get("scales"),
        source_dtype=str(data["source_dtype"]),
        transport_dtype=str(data["transport_dtype"]),
    )


def _part_from_dict(data: dict[str, Any]) -> ArtifactPart:
    return ArtifactPart(
        side=str(data["side"]),
        layer_group_index=int(data["layer_group_index"]),
        layer_indices=tuple(int(x) for x in data["layer_indices"]),
        token_start=int(data["token_start"]),
        token_count=int(data["token_count"]),
        logical_height=int(data["logical_height"]),
        logical_width=int(data["logical_width"]),
        encoded_height=int(data["encoded_height"]),
        encoded_width=int(data["encoded_width"]),
        payload_path=str(data["payload_path"]),
        payload_bytes=int(data["payload_bytes"]),
        checksum=str(data["checksum"]),
        quantization=_part_quant_from_dict(data["quantization"]),
    )


def _variant_from_dict(data: dict[str, Any]) -> ArtifactVariant:
    parts = tuple(sorted((_part_from_dict(part_data) for part_data in data["parts"]), key=lambda part: part.ordering_key))
    estimated_decode_ms = data.get("estimated_decode_ms")
    layout_data = data.get("layout")
    quant_data = data.get("quantization")
    codec_data = data.get("codec")
    return ArtifactVariant(
        name=str(data["name"]),
        parts=parts,
        payload_bytes=int(data.get("payload_bytes", sum(part.payload_bytes for part in parts))),
        estimated_decode_ms=None if estimated_decode_ms is None else float(estimated_decode_ms),
        layout=None
        if layout_data is None
        else LayoutManifest(
            layers_per_frame=int(layout_data["layers_per_frame"]),
            head_rows=int(layout_data["head_rows"]),
            head_cols=int(layout_data["head_cols"]),
            dim_rows=int(layout_data["dim_rows"]),
            dim_cols=int(layout_data["dim_cols"]),
            token_axis_is_time=bool(layout_data.get("token_axis_is_time", True)),
        ),
        quantization=None if quant_data is None else QuantizationManifest(mode=str(quant_data["mode"]), axis=str(quant_data["axis"])),
        codec=None
        if codec_data is None
        else CodecManifest(
            family=str(codec_data["family"]),
            backend=str(codec_data["backend"]),
            lossless_video=bool(codec_data["lossless_video"]),
            codec_params=dict(codec_data.get("codec_params", {})),
        ),
    )


def manifest_from_dict(data: dict[str, Any]) -> ArtifactManifest:
    version = int(data.get("version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest version {version}; expected {SCHEMA_VERSION}")
    kv_shape_data = data["kv_shape"]
    layout_data = data["layout"]
    quant_data = data["quantization"]
    codec_data = data["codec"]
    parts = tuple(sorted((_part_from_dict(part_data) for part_data in data["parts"]), key=lambda part: part.ordering_key))
    variants = tuple(_variant_from_dict(variant_data) for variant_data in data.get("variants", ()))
    return ArtifactManifest(
        version=version,
        source_key=str(data["source_key"]),
        model=str(data["model"]),
        kv_shape=KVShapeManifest(**{key: int(kv_shape_data[key]) for key in ("num_sides", "num_layers", "num_tokens", "num_kv_heads", "head_dim")}),
        layout=LayoutManifest(
            layers_per_frame=int(layout_data["layers_per_frame"]),
            head_rows=int(layout_data["head_rows"]),
            head_cols=int(layout_data["head_cols"]),
            dim_rows=int(layout_data["dim_rows"]),
            dim_cols=int(layout_data["dim_cols"]),
            token_axis_is_time=bool(layout_data.get("token_axis_is_time", True)),
        ),
        quantization=QuantizationManifest(mode=str(quant_data["mode"]), axis=str(quant_data["axis"])),
        codec=CodecManifest(
            family=str(codec_data["family"]),
            backend=str(codec_data["backend"]),
            lossless_video=bool(codec_data["lossless_video"]),
            codec_params=dict(codec_data.get("codec_params", {})),
        ),
        parts=parts,
        variants=variants,
    )
