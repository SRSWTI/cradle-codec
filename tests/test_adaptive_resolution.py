import unittest

from cradle_codec.fetch import select_variant
from cradle_codec.manifest import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactPart,
    ArtifactVariant,
    CodecManifest,
    KVShapeManifest,
    LayoutManifest,
    PartQuantizationManifest,
    QuantizationManifest,
)


def part(path: str, payload_bytes: int) -> ArtifactPart:
    quant = PartQuantizationManifest(
        mode="uint8_minmax",
        axis="channel",
        min_values=[0.0],
        scales=[1.0],
        source_dtype="float32",
        transport_dtype="uint8",
    )
    return ArtifactPart(
        side="k",
        layer_group_index=0,
        layer_indices=(0,),
        token_start=0,
        token_count=1,
        logical_height=1,
        logical_width=1,
        encoded_height=1,
        encoded_width=1,
        payload_path=path,
        payload_bytes=payload_bytes,
        checksum="0" * 64,
        quantization=quant,
    )


def manifest_with_variants(*variants: ArtifactVariant) -> ArtifactManifest:
    return ArtifactManifest(
        version=SCHEMA_VERSION,
        source_key="synthetic/chunk0",
        model="synthetic",
        kv_shape=KVShapeManifest(2, 1, 1, 1, 1),
        layout=LayoutManifest(1, 1, 1, 1, 1),
        quantization=QuantizationManifest("uint8_minmax", "channel"),
        codec=CodecManifest("reference", "raw_reference", True, {}),
        parts=(part("parts/base.bin", 1000),),
        variants=variants,
    )


class AdaptiveResolutionTests(unittest.TestCase):
    def test_legacy_manifest_selects_implicit_base_variant(self) -> None:
        selection = select_variant(manifest_with_variants(), bandwidth_bytes_per_sec=2000.0)

        self.assertEqual(selection.variant.name, "base")
        self.assertEqual(selection.variant.payload_bytes, 1000)
        self.assertEqual(selection.transfer_ms, 500.0)
        self.assertEqual(selection.decode_ms, 0.0)

    def test_selects_lowest_transfer_decode_cost_deterministically(self) -> None:
        low = ArtifactVariant("low", (part("parts/low.bin", 400),), payload_bytes=400, estimated_decode_ms=20.0)
        high = ArtifactVariant("high", (part("parts/high.bin", 800),), payload_bytes=800, estimated_decode_ms=1.0)

        selection = select_variant(manifest_with_variants(low, high), bandwidth_bytes_per_sec=1000.0)

        self.assertEqual(selection.variant.name, "low")
        self.assertEqual(selection.total_estimated_ms, 420.0)

    def test_switch_penalty_can_keep_current_variant(self) -> None:
        low = ArtifactVariant("low", (part("parts/low.bin", 500),), payload_bytes=500, estimated_decode_ms=0.0)
        manifest = manifest_with_variants(low)

        selection = select_variant(
            manifest,
            bandwidth_bytes_per_sec=1000.0,
            current_variant_name="base",
            switch_penalty_ms=600.0,
        )

        self.assertEqual(selection.variant.name, "base")
        self.assertEqual(selection.switch_penalty_ms, 0.0)

    def test_rejects_invalid_bandwidth(self) -> None:
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            select_variant(manifest_with_variants(), bandwidth_bytes_per_sec=0.0)


if __name__ == "__main__":
    unittest.main()
