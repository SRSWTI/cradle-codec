from __future__ import annotations

from dataclasses import dataclass

from cradle_codec.manifest import ArtifactManifest, ArtifactVariant


@dataclass(frozen=True)
class VariantSelection:
    """Deterministic local choice of a manifest variant for a fetch/decode budget."""

    variant: ArtifactVariant
    transfer_ms: float
    decode_ms: float
    switch_penalty_ms: float
    total_estimated_ms: float


def _validate_non_negative(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def estimate_variant_cost_ms(
    variant: ArtifactVariant,
    *,
    bandwidth_bytes_per_sec: float,
    current_variant_name: str | None = None,
    switch_penalty_ms: float = 0.0,
) -> VariantSelection:
    """Estimate fetch plus decode cost for one local artifact variant.

    The estimate is network-free and uses only manifest metadata. Missing decode
    estimates are treated as zero so legacy/base variants remain selectable.
    """

    bandwidth = float(bandwidth_bytes_per_sec)
    if bandwidth <= 0.0:
        raise ValueError("bandwidth_bytes_per_sec must be positive")
    penalty = _validate_non_negative("switch_penalty_ms", switch_penalty_ms)
    decode_ms = 0.0 if variant.estimated_decode_ms is None else _validate_non_negative("estimated_decode_ms", variant.estimated_decode_ms)
    applied_penalty = penalty if current_variant_name is not None and variant.name != current_variant_name else 0.0
    transfer_ms = (float(variant.payload_bytes) / bandwidth) * 1000.0
    return VariantSelection(
        variant=variant,
        transfer_ms=transfer_ms,
        decode_ms=decode_ms,
        switch_penalty_ms=applied_penalty,
        total_estimated_ms=transfer_ms + decode_ms + applied_penalty,
    )


def select_variant(
    manifest: ArtifactManifest,
    *,
    bandwidth_bytes_per_sec: float,
    current_variant_name: str | None = None,
    switch_penalty_ms: float = 0.0,
) -> VariantSelection:
    """Choose the lowest estimated-cost variant advertised by a manifest.

    Manifests without a ``variants`` array expose an implicit ``base`` variant
    backed by the top-level ``parts`` field, preserving single-variant artifacts.
    Ties are resolved deterministically by payload size, then variant name.
    """

    candidates = manifest.variants_with_base()
    if not candidates:
        raise ValueError("manifest does not contain any artifact variants")
    selections = [
        estimate_variant_cost_ms(
            variant,
            bandwidth_bytes_per_sec=bandwidth_bytes_per_sec,
            current_variant_name=current_variant_name,
            switch_penalty_ms=switch_penalty_ms,
        )
        for variant in candidates
    ]
    return min(selections, key=lambda selection: (selection.total_estimated_ms, selection.variant.payload_bytes, selection.variant.name))


__all__ = ["VariantSelection", "estimate_variant_cost_ms", "select_variant"]
