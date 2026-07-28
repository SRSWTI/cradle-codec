"""KV tensor-to-frame layout primitives."""

from .candidates import (
    CandidateEncode,
    CandidateInput,
    LayoutCandidateProfile,
    QWEN3_8B_CANDIDATE_NAMES,
    candidate_name_for_tiling,
    estimate_raw_payload_bytes,
    layout_candidates_for_shape,
    layout_candidates_from_dimensions,
    layout_from_name,
    profile_layout_candidates,
    qwen3_8b_layout_candidates,
    select_layout_candidate,
    validate_candidate_name,
    valid_factor_tilings,
)
from .pack import PackedFrameBatch, pack_kv_to_frame_batches
from .types import FrameBatch, FrameGeometry, HeadDimTiling, KVCodecLayout, KVShape, LayerGroup
from .unpack import restore_frame_batch_into_kv, unpack_frame_batches_to_kv
from .validate import validate_kv_shape, validate_layout, validate_tiling

__all__ = [
    "CandidateEncode",
    "CandidateInput",
    "FrameBatch",
    "FrameGeometry",
    "HeadDimTiling",
    "LayoutCandidateProfile",
    "KVCodecLayout",
    "KVShape",
    "LayerGroup",
    "PackedFrameBatch",
    "QWEN3_8B_CANDIDATE_NAMES",
    "candidate_name_for_tiling",
    "estimate_raw_payload_bytes",
    "layout_candidates_for_shape",
    "layout_candidates_from_dimensions",
    "layout_from_name",
    "pack_kv_to_frame_batches",
    "qwen3_8b_layout_candidates",
    "profile_layout_candidates",
    "restore_frame_batch_into_kv",
    "unpack_frame_batches_to_kv",
    "select_layout_candidate",
    "validate_candidate_name",
    "valid_factor_tilings",
    "validate_kv_shape",
    "validate_layout",
    "validate_tiling",
]
