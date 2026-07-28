from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KVSideName = Literal["k", "v"]


@dataclass(frozen=True)
class KVShape:
    """Canonical logical KV chunk shape: [side, layer, token, kv_head, head_dim]."""

    num_sides: int
    num_layers: int
    num_tokens: int
    num_kv_heads: int
    head_dim: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_sides,
            self.num_layers,
            self.num_tokens,
            self.num_kv_heads,
            self.head_dim,
        )


@dataclass(frozen=True)
class HeadDimTiling:
    """A head-preserving 2D tiling for the [kv_head, head_dim] axes."""

    head_rows: int
    head_cols: int
    dim_rows: int
    dim_cols: int

    @property
    def logical_height(self) -> int:
        return self.head_rows * self.dim_rows

    @property
    def logical_width(self) -> int:
        return self.head_cols * self.dim_cols

    @property
    def name(self) -> str:
        return f"h{self.head_rows}x{self.head_cols}_d{self.dim_rows}x{self.dim_cols}"


@dataclass(frozen=True)
class LayerGroup:
    """A contiguous group of up to layers_per_frame model layers."""

    group_index: int
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class FrameGeometry:
    """Logical and encoded frame dimensions for one frame batch."""

    logical_height: int
    logical_width: int
    encoded_height: int
    encoded_width: int
    channels: int = 3

    @property
    def logical_shape(self) -> tuple[int, int, int]:
        return (self.logical_height, self.logical_width, self.channels)

    @property
    def encoded_shape(self) -> tuple[int, int, int]:
        return (self.encoded_height, self.encoded_width, self.channels)


@dataclass(frozen=True)
class KVCodecLayout:
    """Paper-aligned mapping from KV tensors to RGB/YUV-like video frames."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    layers_per_frame: int
    tiling: HeadDimTiling
    token_axis_is_time: bool = True

    @property
    def name(self) -> str:
        return f"paper_{self.tiling.name}_{self.tiling.logical_height}x{self.tiling.logical_width}"

    @property
    def num_layer_groups(self) -> int:
        return (self.num_layers + self.layers_per_frame - 1) // self.layers_per_frame

    @property
    def geometry(self) -> FrameGeometry:
        return FrameGeometry(
            logical_height=self.tiling.logical_height,
            logical_width=self.tiling.logical_width,
            encoded_height=self.tiling.logical_height,
            encoded_width=self.tiling.logical_width,
            channels=3,
        )

    def layer_groups(self) -> tuple[LayerGroup, ...]:
        groups: list[LayerGroup] = []
        for group_index, start in enumerate(range(0, self.num_layers, self.layers_per_frame)):
            end = min(start + self.layers_per_frame, self.num_layers)
            groups.append(LayerGroup(group_index=group_index, layer_indices=tuple(range(start, end))))
        return tuple(groups)


@dataclass(frozen=True)
class FrameBatch:
    """One side + one layer group packed as token-adjacent video frames."""

    side: int
    side_name: KVSideName
    layer_group: LayerGroup
    token_start: int
    token_count: int
    geometry: FrameGeometry
    # The frame array is intentionally not stored here; keep metadata immutable and arrays external.
