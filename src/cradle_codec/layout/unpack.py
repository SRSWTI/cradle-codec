from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .pack import PackedFrameBatch
from .types import KVCodecLayout
from .validate import validate_layout


def _frame_plane_to_head_dim(plane: np.ndarray, layout: KVCodecLayout) -> np.ndarray:
    """Map [T, head_rows*dim_rows, head_cols*dim_cols] back to [T, H, D]."""

    tiling = layout.tiling
    expected = (tiling.logical_height, tiling.logical_width)
    if plane.shape[1:] != expected:
        raise ValueError(f"expected frame plane tail {expected}, got {plane.shape[1:]}")
    token_count = plane.shape[0]
    reshaped = plane.reshape(
        token_count,
        tiling.head_rows,
        tiling.dim_rows,
        tiling.head_cols,
        tiling.dim_cols,
    )
    return np.ascontiguousarray(reshaped.transpose(0, 1, 3, 2, 4).reshape(
        token_count,
        layout.num_kv_heads,
        layout.head_dim,
    ))



def restore_frame_batch_into_kv(
    batch: PackedFrameBatch,
    layout: KVCodecLayout,
    destination: np.ndarray,
    *,
    num_tokens: int,
) -> None:
    """Restore one decoded frame batch into a canonical KV destination array.

    This is the framewise inverse of one PackedFrameBatch emitted by the packer:
    token frames are restored into ``destination[side, layer, token, head, dim]``
    for the batch side and contiguous layer group. Callers can stream decoded
    parts through this helper and avoid holding every decoded frame batch at once.
    """

    validate_layout(layout)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    expected_destination_shape = (2, layout.num_layers, num_tokens, layout.num_kv_heads, layout.head_dim)
    if tuple(destination.shape) != expected_destination_shape:
        raise ValueError(f"expected destination shape {expected_destination_shape}, got {tuple(destination.shape)}")

    meta = batch.metadata
    frames = batch.frames
    if frames.ndim != 4:
        raise ValueError(f"expected frames rank 4 [T,H,W,3], got rank {frames.ndim}")
    if meta.side not in (0, 1):
        raise ValueError(f"side must be 0 or 1, got {meta.side}")
    if meta.token_start < 0 or meta.token_count <= 0:
        raise ValueError(f"invalid token range start={meta.token_start} count={meta.token_count}")
    token_end = meta.token_start + meta.token_count
    if token_end > num_tokens:
        raise ValueError(f"token range {meta.token_start}:{token_end} exceeds num_tokens={num_tokens}")

    expected_geometry = layout.geometry
    expected_shape = (
        meta.token_count,
        expected_geometry.logical_height,
        expected_geometry.logical_width,
        expected_geometry.channels,
    )
    if tuple(frames.shape) != expected_shape:
        raise ValueError(f"expected frames shape {expected_shape}, got {tuple(frames.shape)}")
    if meta.geometry.logical_height != expected_geometry.logical_height or meta.geometry.logical_width != expected_geometry.logical_width:
        raise ValueError(
            "batch logical geometry does not match layout: "
            f"{meta.geometry.logical_height}x{meta.geometry.logical_width} != "
            f"{expected_geometry.logical_height}x{expected_geometry.logical_width}"
        )

    for channel, layer_index in enumerate(meta.layer_group.layer_indices):
        if layer_index < 0 or layer_index >= layout.num_layers:
            raise ValueError(f"layer index out of range: {layer_index}")
        destination[meta.side, layer_index, meta.token_start:token_end, :, :] = _frame_plane_to_head_dim(
            frames[..., channel], layout
        )


def unpack_frame_batches_to_kv(batches: Sequence[PackedFrameBatch], layout: KVCodecLayout, *, num_tokens: int) -> np.ndarray:
    """Invert pack_kv_to_frame_batches into canonical [2, L, T, H, D] KV layout."""

    validate_layout(layout)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    restored: np.ndarray | None = None
    seen: set[tuple[int, int]] = set()
    expected_geometry = layout.geometry

    for batch in batches:
        meta = batch.metadata
        key = (meta.side, meta.layer_group.group_index)
        if key in seen:
            raise ValueError(f"duplicate frame batch for side={meta.side} group={meta.layer_group.group_index}")
        seen.add(key)
        frames = batch.frames
        if frames.ndim != 4:
            raise ValueError(f"expected frames rank 4 [T,H,W,3], got rank {frames.ndim}")
        expected_shape = (
            num_tokens,
            expected_geometry.logical_height,
            expected_geometry.logical_width,
            expected_geometry.channels,
        )
        if tuple(frames.shape) != expected_shape:
            raise ValueError(f"expected frames shape {expected_shape}, got {tuple(frames.shape)}")
        if meta.side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {meta.side}")
        if meta.token_start != 0 or meta.token_count != num_tokens:
            raise ValueError(
                f"expected full-token batch token_start=0 token_count={num_tokens}, "
                f"got {meta.token_start}/{meta.token_count}"
            )
        if restored is None:
            restored = np.empty(
                (2, layout.num_layers, num_tokens, layout.num_kv_heads, layout.head_dim),
                dtype=frames.dtype,
            )
        restore_frame_batch_into_kv(batch, layout, restored, num_tokens=num_tokens)

    expected_keys = {(side, group.group_index) for side in range(2) for group in layout.layer_groups()}
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"missing frame batches: {sorted(missing)}")
    if restored is None:
        raise ValueError("no frame batches supplied")
    return restored
