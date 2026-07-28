from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Protocol

import numpy as np

from cradle_codec.codec import FrameCodec
from cradle_codec.layout import restore_frame_batch_into_kv
from cradle_codec.manifest import ArtifactManifest, ArtifactPart, verify_part_payload
from cradle_codec.pipeline import PartKey, decode_part_payload, layout_from_manifest, part_key, select_manifest_parts, validate_full_manifest_parts
from cradle_codec.store import LocalArtifactStore


class ArtifactStore(Protocol):
    def artifact_path(self, source_key: str) -> Path:
        ...

    def load_manifest(self, source_key: str) -> ArtifactManifest:
        ...


@dataclass(frozen=True)
class FetchDecodeTiming:
    select_ms: float
    transfer_ms: float
    decode_ms: float
    restore_ms: float
    total_ms: float


@dataclass(frozen=True)
class FetchDecodeResult:
    source_key: str
    artifact_dir: Path
    variant_name: str | None
    part_keys: tuple[PartKey, ...]
    kv: np.ndarray
    timing: FetchDecodeTiming


class LocalFetchDecodeController:
    """Local stand-in for remote KV fetch/decode/restore.

    The controller reads artifacts through LocalArtifactStore paths, but keeps the
    same boundaries a remote fetch pipeline needs: manifest/variant selection,
    per-part transfer with checksum verification, decode/dequantize, and
    framewise restore into a destination KV array.
    """

    def __init__(self, store: ArtifactStore | str | Path, *, codec: FrameCodec | None = None) -> None:
        self.store: ArtifactStore = LocalArtifactStore(store) if isinstance(store, (str, Path)) else store
        self.codec = codec

    def fetch_decode(
        self,
        source_key: str,
        *,
        variant_name: str | None = None,
        part_keys: Iterable[PartKey] | None = None,
        destination: np.ndarray | None = None,
        bandwidth_bytes_per_sec: float | None = None,
        current_variant_name: str | None = None,
        switch_penalty_ms: float = 0.0,
    ) -> FetchDecodeResult:
        """Fetch selected artifact parts, decode them, and restore into canonical KV.

        ``part_keys`` addresses parts as ``(side, layer_group_index)``. Passing
        ``variant_name`` selects a named manifest variant. Passing
        ``bandwidth_bytes_per_sec`` with no explicit variant lets the manifest
        selector choose a variant during the select phase. Full reconstruction
        (the default) allocates the destination lazily using the first decoded
        batch dtype. Partial requests must pass ``destination`` so untouched KV
        slots remain owned by the caller rather than fabricated by the fetch
        layer.
        """

        total_start = perf_counter()
        select_start = perf_counter()
        artifact_dir = self.store.artifact_path(source_key)
        manifest = self.store.load_manifest(source_key)
        resolved_variant_name = variant_name
        if resolved_variant_name is None and bandwidth_bytes_per_sec is not None:
            from .adaptive_resolution import select_variant

            resolved_variant_name = select_variant(
                manifest,
                bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
                current_variant_name=current_variant_name,
                switch_penalty_ms=switch_penalty_ms,
            ).variant.name
        selected_parts = select_manifest_parts(manifest, variant_name=resolved_variant_name, part_keys=part_keys)
        variant_parts = validate_full_manifest_parts(manifest, variant_name=resolved_variant_name)
        selected_keys = tuple(part_key(part) for part in selected_parts)
        selected_key_set = set(selected_keys)
        full_key_set = {part_key(part) for part in variant_parts}
        if selected_key_set != full_key_set and destination is None:
            raise ValueError("partial part fetch requires a destination KV array")
        if not selected_parts:
            raise ValueError("no artifact parts selected for fetch/decode")
        layout = layout_from_manifest(manifest, variant_name=resolved_variant_name)
        num_tokens = manifest.kv_shape.num_tokens
        select_ms = _elapsed_ms(select_start)

        transfer_ms = 0.0
        decode_ms = 0.0
        restore_ms = 0.0
        restored = destination

        for part in selected_parts:
            transfer_start = perf_counter()
            data = verify_part_payload(artifact_dir, part)
            transfer_ms += _elapsed_ms(transfer_start)

            decode_start = perf_counter()
            batch = decode_part_payload(manifest, part, data, codec=self.codec, variant_name=resolved_variant_name)
            decode_ms += _elapsed_ms(decode_start)

            if restored is None:
                restored = np.empty(
                    (2, layout.num_layers, num_tokens, layout.num_kv_heads, layout.head_dim),
                    dtype=batch.frames.dtype,
                )
            restore_start = perf_counter()
            restore_frame_batch_into_kv(batch, layout, restored, num_tokens=num_tokens)
            restore_ms += _elapsed_ms(restore_start)

        if restored is None:
            raise ValueError("no KV array was restored")

        return FetchDecodeResult(
            source_key=source_key,
            artifact_dir=artifact_dir,
            variant_name=resolved_variant_name,
            part_keys=selected_keys,
            kv=restored,
            timing=FetchDecodeTiming(
                select_ms=select_ms,
                transfer_ms=transfer_ms,
                decode_ms=decode_ms,
                restore_ms=restore_ms,
                total_ms=_elapsed_ms(total_start),
            ),
        )


def available_variant_names(manifest: ArtifactManifest) -> tuple[str, ...]:
    return tuple(variant.name for variant in manifest.variants_with_base())


def variant_payload_bytes(manifest: ArtifactManifest, variant_name: str | None = None) -> int:
    parts: tuple[ArtifactPart, ...] = select_manifest_parts(manifest, variant_name=variant_name)
    return sum(part.payload_bytes for part in parts)


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0
