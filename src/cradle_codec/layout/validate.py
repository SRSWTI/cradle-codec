from __future__ import annotations

from .types import FrameGeometry, HeadDimTiling, KVCodecLayout, KVShape


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_kv_shape(shape: KVShape) -> KVShape:
    if shape.num_sides != 2:
        raise ValueError(f"num_sides must be 2 for K/V, got {shape.num_sides}")
    _require_positive("num_layers", shape.num_layers)
    _require_positive("num_tokens", shape.num_tokens)
    _require_positive("num_kv_heads", shape.num_kv_heads)
    _require_positive("head_dim", shape.head_dim)
    return shape


def validate_tiling(tiling: HeadDimTiling, *, num_kv_heads: int, head_dim: int) -> HeadDimTiling:
    _require_positive("head_rows", tiling.head_rows)
    _require_positive("head_cols", tiling.head_cols)
    _require_positive("dim_rows", tiling.dim_rows)
    _require_positive("dim_cols", tiling.dim_cols)
    if tiling.head_rows * tiling.head_cols != num_kv_heads:
        raise ValueError(
            "head tiling must cover kv heads exactly: "
            f"{tiling.head_rows}*{tiling.head_cols} != {num_kv_heads}"
        )
    if tiling.dim_rows * tiling.dim_cols != head_dim:
        raise ValueError(
            "dim tiling must cover head_dim exactly: "
            f"{tiling.dim_rows}*{tiling.dim_cols} != {head_dim}"
        )
    return tiling


def validate_geometry(geometry: FrameGeometry) -> FrameGeometry:
    _require_positive("logical_height", geometry.logical_height)
    _require_positive("logical_width", geometry.logical_width)
    _require_positive("encoded_height", geometry.encoded_height)
    _require_positive("encoded_width", geometry.encoded_width)
    if geometry.channels != 3:
        raise ValueError(f"channels must be 3 for video codec frames, got {geometry.channels}")
    if geometry.encoded_height < geometry.logical_height:
        raise ValueError("encoded_height cannot be smaller than logical_height")
    if geometry.encoded_width < geometry.logical_width:
        raise ValueError("encoded_width cannot be smaller than logical_width")
    return geometry


def validate_layout(layout: KVCodecLayout) -> KVCodecLayout:
    _require_positive("num_layers", layout.num_layers)
    _require_positive("num_kv_heads", layout.num_kv_heads)
    _require_positive("head_dim", layout.head_dim)
    if layout.layers_per_frame not in (1, 2, 3):
        raise ValueError(f"layers_per_frame must be 1, 2, or 3, got {layout.layers_per_frame}")
    if not layout.token_axis_is_time:
        raise ValueError("token_axis_is_time must remain True for paper inter-frame layout")
    validate_tiling(layout.tiling, num_kv_heads=layout.num_kv_heads, head_dim=layout.head_dim)
    expected_height = layout.tiling.head_rows * layout.tiling.dim_rows
    expected_width = layout.tiling.head_cols * layout.tiling.dim_cols
    geometry = layout.geometry
    if geometry.logical_height != expected_height:
        raise ValueError(f"logical_height mismatch: {geometry.logical_height} != {expected_height}")
    if geometry.logical_width != expected_width:
        raise ValueError(f"logical_width mismatch: {geometry.logical_width} != {expected_width}")
    validate_geometry(geometry)
    return layout


def validate_kv_array_shape(actual: tuple[int, ...], expected: KVShape | KVCodecLayout, *, num_tokens: int | None = None) -> None:
    if isinstance(expected, KVShape):
        expected_tuple = expected.as_tuple
    else:
        if num_tokens is None:
            raise ValueError("num_tokens is required when validating against KVCodecLayout")
        expected_tuple = (2, expected.num_layers, num_tokens, expected.num_kv_heads, expected.head_dim)
    if tuple(actual) != expected_tuple:
        raise ValueError(f"expected KV shape {expected_tuple}, got {tuple(actual)}")
