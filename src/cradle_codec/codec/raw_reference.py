from __future__ import annotations

import hashlib
import io

import numpy as np

from cradle_codec.layout import FrameGeometry

from .api import EncodedPayload


def checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RawReferenceCodec:
    """Reversible CPU reference backend for validating the KVCodec pipeline.

    This is not a production video codec. It stores frame arrays in an NPZ container
    so pack/quantize/manifest/reconstruct logic can be tested without GPU, FFmpeg,
    or NVENC/NVDEC availability.
    """

    codec_name = "raw_reference"

    def encode(self, frames: np.ndarray, geometry: FrameGeometry) -> EncodedPayload:
        if frames.ndim != 4:
            raise ValueError(f"expected frames rank 4 [T,H,W,3], got rank {frames.ndim}")
        expected_tail = geometry.logical_shape
        if tuple(frames.shape[1:]) != expected_tail:
            raise ValueError(f"expected frame tail {expected_tail}, got {tuple(frames.shape[1:])}")
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            frames=np.ascontiguousarray(frames),
            logical_height=np.array(geometry.logical_height, dtype=np.int64),
            logical_width=np.array(geometry.logical_width, dtype=np.int64),
            channels=np.array(geometry.channels, dtype=np.int64),
        )
        data = buffer.getvalue()
        return EncodedPayload(
            codec_name=self.codec_name,
            data=data,
            payload_bytes=len(data),
            checksum=checksum_bytes(data),
        )

    def decode(self, payload: EncodedPayload | bytes, geometry: FrameGeometry) -> np.ndarray:
        data = payload.data if isinstance(payload, EncodedPayload) else payload
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
                frames = np.ascontiguousarray(loaded["frames"])
                logical_height = int(loaded["logical_height"])
                logical_width = int(loaded["logical_width"])
                channels = int(loaded["channels"])
        except Exception as exc:  # noqa: BLE001 - expose a clear codec boundary error.
            raise ValueError("failed to decode raw_reference payload") from exc
        if (logical_height, logical_width, channels) != geometry.logical_shape:
            raise ValueError(
                "payload geometry mismatch: "
                f"{(logical_height, logical_width, channels)} != {geometry.logical_shape}"
            )
        expected_tail = geometry.logical_shape
        if frames.ndim != 4 or tuple(frames.shape[1:]) != expected_tail:
            raise ValueError(f"decoded frame shape mismatch: got {frames.shape}, expected tail {expected_tail}")
        return frames
