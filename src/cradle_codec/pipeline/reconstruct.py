from __future__ import annotations

from pathlib import Path

import numpy as np

from .decode_chunk import decode_artifact_parts, decode_kv_artifact, layout_from_manifest


def reconstruct_kv(artifact_dir: str | Path) -> np.ndarray:
    return decode_kv_artifact(artifact_dir)


__all__ = ["decode_artifact_parts", "decode_kv_artifact", "layout_from_manifest", "reconstruct_kv"]
