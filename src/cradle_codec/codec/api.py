from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cradle_codec.layout import FrameGeometry


@dataclass(frozen=True)
class EncodedPayload:
    codec_name: str
    data: bytes
    payload_bytes: int
    checksum: str


class FrameCodec(Protocol):
    codec_name: str

    def encode(self, frames: np.ndarray, geometry: FrameGeometry) -> EncodedPayload:
        ...

    def decode(self, payload: EncodedPayload | bytes, geometry: FrameGeometry) -> np.ndarray:
        ...
