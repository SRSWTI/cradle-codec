from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

QuantizationMode = Literal["uint8_minmax"]
QuantizationAxis = Literal["frame", "channel", "part"]


@dataclass(frozen=True)
class QuantizationSpec:
    mode: QuantizationMode
    axis: QuantizationAxis = "part"


@dataclass(frozen=True)
class QuantizationMetadata:
    mode: str
    axis: str
    min_values: np.ndarray | None
    scales: np.ndarray | None
    source_dtype: str
    transport_dtype: str
