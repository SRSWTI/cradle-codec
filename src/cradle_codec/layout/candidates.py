from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pack import pack_kv_to_frame_batches
from .types import FrameGeometry, HeadDimTiling, KVCodecLayout, KVShape
from .validate import validate_kv_shape, validate_layout, validate_tiling

_CANDIDATE_RE = re.compile(r"^(?:paper_)?h(?P<hr>\d+)x(?P<hc>\d+)_d(?P<dr>\d+)x(?P<dc>\d+)(?:_(?P<gh>\d+)x(?P<gw>\d+))?$")

CandidateInput = str | HeadDimTiling | KVCodecLayout
CandidateEncode = Callable[[np.ndarray, FrameGeometry], object]


@dataclass(frozen=True)
class LayoutCandidateProfile:
    """Payload-size score for one deterministic layout candidate."""

    name: str
    layout: KVCodecLayout
    raw_payload_bytes: int
    encoded_payload_bytes: int
    part_count: int
    codec_name: str | None = None

    @property
    def encoded_to_raw_ratio(self) -> float:
        return self.encoded_payload_bytes / self.raw_payload_bytes


QWEN3_8B_CANDIDATE_NAMES: tuple[str, ...] = (
    "h1x8_d32x4_32x32",
    "h1x8_d16x8_16x64",
    "h1x8_d8x16_8x128",
    "h2x4_d16x8_32x32",
    "h4x2_d8x16_32x32",
)


def candidate_name_for_tiling(tiling: HeadDimTiling, *, paper_prefix: bool = False) -> str:
    """Return the canonical paper-style candidate name with explicit frame geometry."""

    prefix = "paper_" if paper_prefix else ""
    return f"{prefix}{tiling.name}_{tiling.logical_height}x{tiling.logical_width}"


def _parse_candidate_name(name: str) -> HeadDimTiling:
    match = _CANDIDATE_RE.match(name)
    if match is None:
        raise ValueError(f"invalid layout candidate name: {name!r}")
    tiling = HeadDimTiling(
        head_rows=int(match.group("hr")),
        head_cols=int(match.group("hc")),
        dim_rows=int(match.group("dr")),
        dim_cols=int(match.group("dc")),
    )
    suffix_height = match.group("gh")
    suffix_width = match.group("gw")
    if suffix_height is not None and suffix_width is not None:
        expected = (tiling.logical_height, tiling.logical_width)
        actual = (int(suffix_height), int(suffix_width))
        if actual != expected:
            raise ValueError(f"candidate geometry suffix mismatch: {actual[0]}x{actual[1]} != {expected[0]}x{expected[1]}")
    return tiling


def tiling_from_name(name: str) -> HeadDimTiling:
    return _parse_candidate_name(name)


def validate_candidate_name(
    name: str,
    *,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> HeadDimTiling:
    """Parse and validate a candidate name against optional model dimensions."""

    if (num_kv_heads is None) != (head_dim is None):
        raise ValueError("num_kv_heads and head_dim must be provided together")
    tiling = tiling_from_name(name)
    if num_kv_heads is not None and head_dim is not None:
        validate_tiling(tiling, num_kv_heads=num_kv_heads, head_dim=head_dim)
    return tiling


def layout_from_name(
    name: str,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    layers_per_frame: int = 3,
) -> KVCodecLayout:
    layout = KVCodecLayout(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        layers_per_frame=layers_per_frame,
        tiling=validate_candidate_name(name, num_kv_heads=num_kv_heads, head_dim=head_dim),
    )
    return validate_layout(layout)


def qwen3_8b_layout_candidates(*, num_layers: int = 36, layers_per_frame: int = 3) -> tuple[KVCodecLayout, ...]:
    return tuple(
        layout_from_name(
            name,
            num_layers=num_layers,
            num_kv_heads=8,
            head_dim=128,
            layers_per_frame=layers_per_frame,
        )
        for name in QWEN3_8B_CANDIDATE_NAMES
    )


def valid_factor_tilings(num_kv_heads: int, head_dim: int) -> Iterable[HeadDimTiling]:
    """Generate head-preserving geometric tilings without permuting heads/dims."""

    for head_rows in range(1, num_kv_heads + 1):
        if num_kv_heads % head_rows != 0:
            continue
        head_cols = num_kv_heads // head_rows
        for dim_rows in range(1, head_dim + 1):
            if head_dim % dim_rows != 0:
                continue
            dim_cols = head_dim // dim_rows
            yield HeadDimTiling(head_rows, head_cols, dim_rows, dim_cols)


def layout_candidates_for_shape(
    shape: KVShape,
    *,
    layers_per_frame: int = 3,
    candidate_names: Iterable[str] | None = None,
) -> tuple[KVCodecLayout, ...]:
    """Generate deterministic valid paper-style layout candidates for a KV shape."""

    shape = validate_kv_shape(shape)
    if candidate_names is None:
        tilings = tuple(valid_factor_tilings(shape.num_kv_heads, shape.head_dim))
    else:
        tilings = tuple(
            validate_candidate_name(name, num_kv_heads=shape.num_kv_heads, head_dim=shape.head_dim)
            for name in candidate_names
        )
    return tuple(
        validate_layout(
            KVCodecLayout(
                num_layers=shape.num_layers,
                num_kv_heads=shape.num_kv_heads,
                head_dim=shape.head_dim,
                layers_per_frame=layers_per_frame,
                tiling=tiling,
            )
        )
        for tiling in tilings
    )


def layout_candidates_from_dimensions(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_tokens: int = 1,
    layers_per_frame: int = 3,
) -> tuple[KVCodecLayout, ...]:
    """Generate candidates from model dimensions when a full KVShape is not available."""

    shape = KVShape(
        num_sides=2,
        num_layers=num_layers,
        num_tokens=num_tokens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    return layout_candidates_for_shape(shape, layers_per_frame=layers_per_frame)


def estimate_raw_payload_bytes(shape: KVShape, layout: KVCodecLayout, *, dtype: np.dtype[Any] | type[Any] = np.uint8) -> int:
    """Estimate unencoded packed-frame bytes for a shape/layout pair."""

    shape = validate_kv_shape(shape)
    layout = validate_layout(layout)
    if (layout.num_layers, layout.num_kv_heads, layout.head_dim) != (shape.num_layers, shape.num_kv_heads, shape.head_dim):
        raise ValueError("layout dimensions must match KV shape")
    geometry = layout.geometry
    per_part = shape.num_tokens * geometry.logical_height * geometry.logical_width * geometry.channels * np.dtype(dtype).itemsize
    return per_part * 2 * layout.num_layer_groups


def _shape_from_kv(kv: np.ndarray) -> KVShape:
    if kv.ndim != 5:
        raise ValueError(f"expected KV rank 5 [2,L,T,H,D], got rank {kv.ndim}")
    return validate_kv_shape(
        KVShape(
            num_sides=int(kv.shape[0]),
            num_layers=int(kv.shape[1]),
            num_tokens=int(kv.shape[2]),
            num_kv_heads=int(kv.shape[3]),
            head_dim=int(kv.shape[4]),
        )
    )


def _coerce_candidate(candidate: CandidateInput, shape: KVShape, *, layers_per_frame: int) -> KVCodecLayout:
    if isinstance(candidate, KVCodecLayout):
        layout = validate_layout(candidate)
    elif isinstance(candidate, HeadDimTiling):
        layout = validate_layout(
            KVCodecLayout(
                num_layers=shape.num_layers,
                num_kv_heads=shape.num_kv_heads,
                head_dim=shape.head_dim,
                layers_per_frame=layers_per_frame,
                tiling=validate_tiling(candidate, num_kv_heads=shape.num_kv_heads, head_dim=shape.head_dim),
            )
        )
    elif isinstance(candidate, str):
        layout = layout_from_name(
            candidate,
            num_layers=shape.num_layers,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            layers_per_frame=layers_per_frame,
        )
    else:
        raise TypeError(f"unsupported candidate type: {type(candidate).__name__}")
    if (layout.num_layers, layout.num_kv_heads, layout.head_dim) != (shape.num_layers, shape.num_kv_heads, shape.head_dim):
        raise ValueError("candidate layout dimensions must match KV tensor shape")
    return layout


def _payload_size(result: object) -> int:
    payload_bytes = getattr(result, "payload_bytes", None)
    if payload_bytes is not None:
        size = int(payload_bytes)
    elif isinstance(result, int):
        size = result
    elif isinstance(result, (bytes, bytearray, memoryview)):
        size = len(result)
    else:
        data = getattr(result, "data", None)
        if isinstance(data, (bytes, bytearray, memoryview)):
            size = len(data)
        else:
            raise TypeError("encode callable must return payload_bytes, int, bytes, bytearray, or memoryview")
    if size < 0:
        raise ValueError(f"encoded payload size must be non-negative, got {size}")
    return size


def _default_reference_encoder() -> tuple[CandidateEncode, str]:
    from cradle_codec.codec import RawReferenceCodec

    codec = RawReferenceCodec()
    return codec.encode, codec.codec_name


def profile_layout_candidates(
    kv: np.ndarray,
    candidates: Iterable[CandidateInput] | None = None,
    *,
    layers_per_frame: int = 3,
    encode: CandidateEncode | None = None,
    codec: Any | None = None,
) -> tuple[LayoutCandidateProfile, ...]:
    """Score candidates by encoded and raw packed-frame payload bytes.

    The default encoder is the CPU-only RawReferenceCodec. A codec object follows the
    same payload_geometry hook as the artifact encoder; tests and experiments may
    inject a lightweight encode callable returning an EncodedPayload-like object, an
    integer byte count, or bytes.
    """

    if encode is not None and codec is not None:
        raise ValueError("provide either encode or codec, not both")
    shape = _shape_from_kv(kv)
    if candidates is None:
        layouts = layout_candidates_for_shape(shape, layers_per_frame=layers_per_frame)
    else:
        layouts = tuple(_coerce_candidate(candidate, shape, layers_per_frame=layers_per_frame) for candidate in candidates)
    geometry_for_encode: Callable[[FrameGeometry], FrameGeometry] = lambda geometry: geometry
    if encode is None:
        if codec is None:
            encode, codec_name = _default_reference_encoder()
        else:
            encode = codec.encode
            adjust_geometry = getattr(codec, "payload_geometry", None)
            if callable(adjust_geometry):
                geometry_for_encode = adjust_geometry
            codec_name = getattr(codec, "codec_name", type(codec).__name__)
    else:
        codec_name = getattr(encode, "__name__", "injected")

    profiles: list[LayoutCandidateProfile] = []
    for layout in layouts:
        batches = pack_kv_to_frame_batches(kv, layout)
        raw_payload_bytes = sum(int(batch.frames.nbytes) for batch in batches)
        encoded_payload_bytes = sum(
            _payload_size(encode(batch.frames, geometry_for_encode(batch.metadata.geometry))) for batch in batches
        )
        profiles.append(
            LayoutCandidateProfile(
                name=candidate_name_for_tiling(layout.tiling),
                layout=layout,
                raw_payload_bytes=raw_payload_bytes,
                encoded_payload_bytes=encoded_payload_bytes,
                part_count=len(batches),
                codec_name=codec_name,
            )
        )
    return tuple(sorted(profiles, key=lambda profile: (profile.encoded_payload_bytes, profile.raw_payload_bytes, profile.name)))


def select_layout_candidate(
    kv: np.ndarray,
    candidates: Iterable[CandidateInput] | None = None,
    *,
    layers_per_frame: int = 3,
    encode: CandidateEncode | None = None,
    codec: Any | None = None,
) -> LayoutCandidateProfile:
    """Return the lowest-encoded-byte candidate profile with deterministic ties."""

    profiles = profile_layout_candidates(kv, candidates, layers_per_frame=layers_per_frame, encode=encode, codec=codec)
    if not profiles:
        raise ValueError("no layout candidates to select from")
    return profiles[0]
