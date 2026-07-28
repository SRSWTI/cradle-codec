from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class CacheAxisOrder:
    """Axis positions for a caller-provided paged KV cache array."""

    side_axis: int = 0
    layer_axis: int = 1
    slot_axis: int = 2
    kv_head_axis: int = 3
    head_dim_axis: int = 4

    def __post_init__(self) -> None:
        axes = (self.side_axis, self.layer_axis, self.slot_axis, self.kv_head_axis, self.head_dim_axis)
        if sorted(axes) != [0, 1, 2, 3, 4]:
            raise ValueError("cache axes must be a permutation of 0..4")

    @property
    def source_axes(self) -> tuple[int, int, int, int, int]:
        return (self.side_axis, self.layer_axis, self.slot_axis, self.kv_head_axis, self.head_dim_axis)


@dataclass(frozen=True, slots=True)
class PagedRestorePlan:
    """Map logical restored token positions into physical cache slots.

    ``slot_indices`` contains one destination slot per restored token, starting at
    ``token_start`` in the canonical restored KV chunk shape
    ``[side, layer, token, kv_head, head_dim]``.
    """

    slot_indices: tuple[int, ...]
    token_start: int = 0
    side_indices: tuple[int, ...] = (0, 1)
    layer_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        slots = tuple(int(slot) for slot in self.slot_indices)
        sides = tuple(int(side) for side in self.side_indices)
        layers = None if self.layer_indices is None else tuple(int(layer) for layer in self.layer_indices)
        object.__setattr__(self, "slot_indices", slots)
        object.__setattr__(self, "side_indices", sides)
        object.__setattr__(self, "layer_indices", layers)
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if not slots:
            raise ValueError("slot_indices must not be empty")
        if len(set(slots)) != len(slots):
            raise ValueError("slot_indices must be unique")
        if any(slot < 0 for slot in slots):
            raise ValueError("slot_indices must be non-negative")
        if not sides or any(side not in (0, 1) for side in sides):
            raise ValueError("side_indices must contain side 0 and/or side 1")
        if len(set(sides)) != len(sides):
            raise ValueError("side_indices must be unique")
        if layers is not None:
            if not layers:
                raise ValueError("layer_indices must not be empty when provided")
            if len(set(layers)) != len(layers):
                raise ValueError("layer_indices must be unique")
            if any(layer < 0 for layer in layers):
                raise ValueError("layer_indices must be non-negative")

    @classmethod
    def from_slots(
        cls,
        slot_indices: Iterable[int],
        *,
        token_start: int = 0,
        side_indices: Iterable[int] = (0, 1),
        layer_indices: Iterable[int] | None = None,
    ) -> "PagedRestorePlan":
        return cls(
            slot_indices=tuple(slot_indices),
            token_start=token_start,
            side_indices=tuple(side_indices),
            layer_indices=None if layer_indices is None else tuple(layer_indices),
        )


@dataclass(frozen=True, slots=True)
class RestoreResult:
    tokens_restored: int
    slots_written: tuple[int, ...]
    sides_written: tuple[int, ...]
    layers_written: tuple[int, ...]


class PagedRestoreAdapter(Protocol):
    def restore(self, restored_kv: np.ndarray, plan: PagedRestorePlan) -> RestoreResult:
        """Write a canonical restored KV chunk into paged cache slots."""


class NumpyPagedKVCacheAdapter:
    """Paged restore adapter backed by a caller-owned NumPy cache array.

    The source KV must use the project canonical layout
    ``[side=2, layer, token, kv_head, head_dim]``.  The destination cache is a
    slot-addressable array with the same logical axes except that token is
    replaced by physical slot; ``CacheAxisOrder`` allows callers to keep their own
    axis order without copying the whole cache.
    """

    def __init__(self, cache: np.ndarray, axes: CacheAxisOrder | None = None) -> None:
        if cache.ndim != 5:
            raise ValueError("paged cache must be a 5-D array")
        self.cache = cache
        self.axes = axes or CacheAxisOrder()

    def _canonical_cache_view(self) -> np.ndarray:
        return np.moveaxis(self.cache, self.axes.source_axes, (0, 1, 2, 3, 4))

    def restore(self, restored_kv: np.ndarray, plan: PagedRestorePlan) -> RestoreResult:
        if restored_kv.ndim != 5:
            raise ValueError("restored_kv must have shape [side, layer, token, kv_head, head_dim]")
        if restored_kv.shape[0] != 2:
            raise ValueError("restored_kv side axis must have length 2")
        cache = self._canonical_cache_view()
        if cache.shape[0] < 2:
            raise ValueError("paged cache side axis must have length at least 2")
        if cache.shape[3:] != restored_kv.shape[3:]:
            raise ValueError("paged cache kv_head/head_dim axes do not match restored_kv")

        token_end = plan.token_start + len(plan.slot_indices)
        if token_end > restored_kv.shape[2]:
            raise ValueError("restore plan token span exceeds restored_kv token axis")
        if any(slot >= cache.shape[2] for slot in plan.slot_indices):
            raise ValueError("restore plan references a slot outside the paged cache")
        if any(side >= restored_kv.shape[0] or side >= cache.shape[0] for side in plan.side_indices):
            raise ValueError("restore plan references a side outside the cache")

        layers = tuple(range(restored_kv.shape[1])) if plan.layer_indices is None else plan.layer_indices
        if cache.shape[1] < restored_kv.shape[1]:
            raise ValueError("paged cache has fewer layers than restored_kv")
        if any(layer >= restored_kv.shape[1] or layer >= cache.shape[1] for layer in layers):
            raise ValueError("restore plan references a layer outside the cache")

        for token_offset, slot in enumerate(plan.slot_indices):
            source_token = plan.token_start + token_offset
            for side in plan.side_indices:
                for layer in layers:
                    cache[side, layer, slot, :, :] = restored_kv[side, layer, source_token, :, :]

        return RestoreResult(
            tokens_restored=len(plan.slot_indices),
            slots_written=plan.slot_indices,
            sides_written=plan.side_indices,
            layers_written=layers,
        )


def restore_to_paged_cache(
    cache: np.ndarray,
    restored_kv: np.ndarray,
    plan: PagedRestorePlan,
    *,
    axes: CacheAxisOrder | None = None,
) -> RestoreResult:
    return NumpyPagedKVCacheAdapter(cache, axes=axes).restore(restored_kv, plan)
