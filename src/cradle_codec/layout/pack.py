from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .types import FrameBatch, KVCodecLayout, KVShape
from .validate import validate_kv_array_shape, validate_kv_shape, validate_layout

_SIDE_NAMES = ("k", "v")


@dataclass(frozen=True)
class PackedFrameBatch:
    """Packed frames plus metadata for one side/layer-group part."""

    metadata: FrameBatch
    frames: np.ndarray


def _head_dim_to_frame_plane(values: np.ndarray, layout: KVCodecLayout) -> np.ndarray:
    """Map [T, H, D] to [T, head_rows*dim_rows, head_cols*dim_cols]."""

    if values.shape[1:] != (layout.num_kv_heads, layout.head_dim):
        raise ValueError(
            f"expected token/head/dim tail {(layout.num_kv_heads, layout.head_dim)}, got {values.shape[1:]}"
        )
    tiling = layout.tiling
    token_count = values.shape[0]
    reshaped = values.reshape(
        token_count,
        tiling.head_rows,
        tiling.head_cols,
        tiling.dim_rows,
        tiling.dim_cols,
    )
    # Preserve head order and within-head dimension order. Only geometric tiling changes.
    return np.ascontiguousarray(reshaped.transpose(0, 1, 3, 2, 4).reshape(
        token_count,
        tiling.logical_height,
        tiling.logical_width,
    ))


def pack_kv_to_frame_batches(kv: np.ndarray, layout: KVCodecLayout) -> list[PackedFrameBatch]:
    """Pack canonical KV [2, L, T, H, D] into paper-style token-time frame batches.

    For each K/V side and contiguous layer group, token index becomes frame index,
    layer index inside the group becomes channel 0/1/2, and [head, dim] is tiled into
    the 2D image plane without mixing heads or reordering values within a head.
    """

    validate_layout(layout)
    if kv.ndim != 5:
        raise ValueError(f"expected KV rank 5 [2,L,T,H,D], got rank {kv.ndim}")
    shape = KVShape(
        num_sides=int(kv.shape[0]),
        num_layers=int(kv.shape[1]),
        num_tokens=int(kv.shape[2]),
        num_kv_heads=int(kv.shape[3]),
        head_dim=int(kv.shape[4]),
    )
    validate_kv_shape(shape)
    validate_kv_array_shape(kv.shape, layout, num_tokens=shape.num_tokens)

    batches: list[PackedFrameBatch] = []
    geometry = layout.geometry
    for side in range(2):
        for layer_group in layout.layer_groups():
            frames = np.zeros(
                (shape.num_tokens, geometry.logical_height, geometry.logical_width, geometry.channels),
                dtype=kv.dtype,
            )
            for channel, layer_index in enumerate(layer_group.layer_indices):
                frames[..., channel] = _head_dim_to_frame_plane(kv[side, layer_index, :, :, :], layout)
            batches.append(
                PackedFrameBatch(
                    metadata=FrameBatch(
                        side=side,
                        side_name=_SIDE_NAMES[side],
                        layer_group=layer_group,
                        token_start=0,
                        token_count=shape.num_tokens,
                        geometry=geometry,
                    ),
                    frames=frames,
                )
            )
    return batches


def iter_expected_batches(layout: KVCodecLayout, *, num_tokens: int) -> Iterable[FrameBatch]:
    validate_layout(layout)
    geometry = layout.geometry
    for side in range(2):
        for layer_group in layout.layer_groups():
            yield FrameBatch(
                side=side,
                side_name=_SIDE_NAMES[side],
                layer_group=layer_group,
                token_start=0,
                token_count=num_tokens,
                geometry=geometry,
            )
