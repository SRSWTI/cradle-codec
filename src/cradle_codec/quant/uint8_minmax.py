from __future__ import annotations

import numpy as np

from .types import QuantizationMetadata


def _axes_for(values: np.ndarray, axis: str) -> tuple[int, ...]:
    if axis == "part":
        return tuple(range(values.ndim))
    if axis == "frame":
        if values.ndim < 1:
            raise ValueError("frame axis requires at least rank-1 input")
        return tuple(range(1, values.ndim))
    if axis == "channel":
        if values.ndim < 1:
            raise ValueError("channel axis requires at least rank-1 input")
        return tuple(range(values.ndim - 1))
    raise ValueError(f"unsupported uint8_minmax axis: {axis!r}")


def quantize_uint8_minmax(values: np.ndarray, *, axis: str = "channel") -> tuple[np.ndarray, QuantizationMetadata]:
    """Quantize floating values to uint8 using min/max metadata.

    This is intentionally lossy. Metadata carries enough min/scale values for a
    deterministic approximate reconstruction, but callers must validate model quality.
    """

    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"uint8_minmax requires floating input, got {values.dtype}")
    values32 = np.asarray(values, dtype=np.float32)
    reduce_axes = _axes_for(values32, axis)
    min_values = values32.min(axis=reduce_axes, keepdims=True)
    max_values = values32.max(axis=reduce_axes, keepdims=True)
    scales = (max_values - min_values) / np.float32(255.0)
    safe_scales = np.where(scales == 0.0, np.float32(1.0), scales)
    quantized = np.rint((values32 - min_values) / safe_scales).clip(0, 255).astype(np.uint8)
    metadata = QuantizationMetadata(
        mode="uint8_minmax",
        axis=axis,
        min_values=np.ascontiguousarray(min_values.squeeze()),
        scales=np.ascontiguousarray(scales.squeeze()),
        source_dtype=str(values.dtype),
        transport_dtype="uint8",
    )
    return np.ascontiguousarray(quantized), metadata


def _restore_shape(values_shape: tuple[int, ...], axis: str) -> tuple[int, ...]:
    if axis == "part":
        return (1,) * len(values_shape)
    if axis == "frame":
        return (values_shape[0],) + (1,) * (len(values_shape) - 1)
    if axis == "channel":
        return (1,) * (len(values_shape) - 1) + (values_shape[-1],)
    raise ValueError(f"unsupported uint8_minmax axis: {axis!r}")


def dequantize_uint8_minmax(values: np.ndarray, metadata: QuantizationMetadata) -> np.ndarray:
    if values.dtype != np.uint8:
        raise TypeError(f"expected uint8 transport, got {values.dtype}")
    if metadata.mode != "uint8_minmax":
        raise ValueError(f"expected uint8_minmax metadata, got {metadata.mode!r}")
    if metadata.min_values is None or metadata.scales is None:
        raise ValueError("uint8_minmax metadata requires min_values and scales")
    min_values = np.asarray(metadata.min_values, dtype=np.float32)
    scales = np.asarray(metadata.scales, dtype=np.float32)
    min_values = min_values.reshape(_restore_shape(values.shape, metadata.axis))
    scales = scales.reshape(_restore_shape(values.shape, metadata.axis))
    reconstructed = values.astype(np.float32) * scales + min_values
    return reconstructed.astype(np.dtype(metadata.source_dtype), copy=False)
