from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_similarity: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "cosine_similarity": self.cosine_similarity,
        }


def compute_error_metrics(reference: np.ndarray, reconstructed: np.ndarray, *, cosine: bool = True) -> ErrorMetrics:
    if reference.shape != reconstructed.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {reconstructed.shape}")
    ref = np.asarray(reference, dtype=np.float64)
    rec = np.asarray(reconstructed, dtype=np.float64)
    diff = rec - ref
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    cosine_similarity: float | None = None
    if cosine:
        ref_flat = ref.reshape(-1)
        rec_flat = rec.reshape(-1)
        denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(rec_flat))
        cosine_similarity = None if denom == 0.0 else float(np.dot(ref_flat, rec_flat) / denom)
    return ErrorMetrics(
        max_abs_error=float(abs_diff.max(initial=0.0)),
        mean_abs_error=float(abs_diff.mean() if abs_diff.size else 0.0),
        rmse=rmse,
        cosine_similarity=cosine_similarity,
    )
