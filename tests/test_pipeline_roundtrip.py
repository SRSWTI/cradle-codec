from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.manifest import ArtifactVariant, read_manifest, write_manifest
from cradle_codec.pipeline import decode_kv_artifact, encode_kv_chunk
from cradle_codec.quant import QuantizationSpec, compute_error_metrics


def synthetic_fp16_kv() -> np.ndarray:
    values = np.arange(2 * 4 * 5 * 2 * 3, dtype=np.float32).reshape(2, 4, 5, 2, 3)
    return (values / 17.0 - 3.0).astype(np.float16)


class PipelineRoundtripTests(unittest.TestCase):
    def layout(self) -> KVCodecLayout:
        return KVCodecLayout(
            num_layers=4,
            num_kv_heads=2,
            head_dim=3,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=2, dim_rows=1, dim_cols=3),
        )


    def test_uint8_minmax_pipeline_roundtrip_reports_bounded_error(self) -> None:
        kv = synthetic_fp16_kv().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = encode_kv_chunk(
                kv,
                tmp,
                source_key="synthetic/chunk0",
                model="synthetic",
                layout=self.layout(),
                quantization=QuantizationSpec("uint8_minmax", "channel"),
            )
            restored = decode_kv_artifact(tmp)
            loaded = read_manifest(Path(tmp) / "manifest.json")

        metrics = compute_error_metrics(kv, restored)
        self.assertEqual(len(manifest.parts), 4)
        self.assertEqual(loaded.kv_shape.num_tokens, 5)
        self.assertEqual(loaded.quantization.mode, "uint8_minmax")
        self.assertEqual([variant.name for variant in loaded.variants], ["base"])
        self.assertEqual(loaded.variant("base").payload_bytes, sum(part.payload_bytes for part in loaded.parts))
        self.assertEqual(restored.shape, kv.shape)
        self.assertLess(metrics.max_abs_error, 0.05)
        self.assertGreater(metrics.cosine_similarity or 0.0, 0.9999)

    def test_decode_defaults_to_top_level_parts_when_variants_exist(self) -> None:
        kv = synthetic_fp16_kv().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = encode_kv_chunk(kv, tmp, source_key="synthetic/chunk0", model="synthetic", layout=self.layout())
            missing_part = replace(manifest.sorted_parts()[0], payload_path="parts/missing-variant.bin")
            write_manifest(
                Path(tmp) / "manifest.json",
                replace(
                    manifest,
                    variants=(ArtifactVariant("broken-low", (missing_part,), missing_part.payload_bytes, estimated_decode_ms=1.0),),
                ),
            )

            restored = decode_kv_artifact(tmp)

        self.assertEqual(restored.shape, kv.shape)

    def test_decode_rejects_layer_indices_that_do_not_match_layout(self) -> None:
        kv = synthetic_fp16_kv()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = encode_kv_chunk(kv, tmp, source_key="synthetic/chunk0", model="synthetic", layout=self.layout())
            malformed = replace(manifest.parts[0], layer_indices=(3,))
            write_manifest(Path(tmp) / "manifest.json", replace(manifest, parts=(malformed, *manifest.parts[1:]), variants=()))

            with self.assertRaisesRegex(ValueError, "layer indices"):
                decode_kv_artifact(tmp)

    def test_decode_fails_on_missing_payload(self) -> None:
        kv = synthetic_fp16_kv()
        with tempfile.TemporaryDirectory() as tmp:
            encode_kv_chunk(kv, tmp, source_key="synthetic/chunk0", model="synthetic", layout=self.layout())
            (Path(tmp) / "parts" / "k.g0.bin").unlink()

            with self.assertRaises(FileNotFoundError):
                decode_kv_artifact(tmp)


if __name__ == "__main__":
    unittest.main()
