import tempfile
import unittest
from dataclasses import replace

import numpy as np

from cradle_codec.fetch import LocalFetchDecodeController, available_variant_names
from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.manifest import ArtifactVariant, write_manifest
from cradle_codec.pipeline import encode_kv_chunk
from cradle_codec.store import LocalArtifactStore


def synthetic_kv() -> np.ndarray:
    values = np.arange(2 * 4 * 5 * 2 * 3, dtype=np.float32).reshape(2, 4, 5, 2, 3)
    return (values / 19.0 - 2.0).astype(np.float32)


class LocalFetchDecodeControllerTests(unittest.TestCase):
    def layout(self) -> KVCodecLayout:
        return KVCodecLayout(
            num_layers=4,
            num_kv_heads=2,
            head_dim=3,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=2, dim_rows=1, dim_cols=3),
        )

    def write_artifact(self, root: str, source_key: str, kv: np.ndarray):
        store = LocalArtifactStore(root)
        artifact_dir = store.artifact_path(source_key)
        manifest = encode_kv_chunk(kv, artifact_dir, source_key=source_key, model="synthetic", layout=self.layout())
        return store, artifact_dir, manifest

    def test_fetch_decode_restores_full_kv_with_manifest_part_verification(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            self.write_artifact(tmp, source_key, kv)
            result = LocalFetchDecodeController(tmp).fetch_decode(source_key)

        self.assertEqual(result.source_key, source_key)
        self.assertEqual(result.variant_name, None)
        self.assertEqual(result.kv.shape, kv.shape)
        self.assertEqual(len(result.part_keys), 4)
        self.assertGreaterEqual(result.timing.select_ms, 0.0)
        self.assertGreaterEqual(result.timing.transfer_ms, 0.0)
        self.assertGreaterEqual(result.timing.decode_ms, 0.0)
        self.assertGreaterEqual(result.timing.restore_ms, 0.0)
        self.assertGreaterEqual(result.timing.total_ms, 0.0)
        np.testing.assert_allclose(result.kv, kv, atol=0.05)

    def test_fetch_decode_selects_manifest_variant(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            _store, artifact_dir, manifest = self.write_artifact(tmp, source_key, kv)
            compact = ArtifactVariant(
                name="compact",
                parts=manifest.sorted_parts(),
                payload_bytes=1,
                estimated_decode_ms=0.0,
            )
            write_manifest(artifact_dir / "manifest.json", replace(manifest, variants=(compact,)))

            controller = LocalFetchDecodeController(tmp)
            result = controller.fetch_decode(source_key, bandwidth_bytes_per_sec=1_000_000.0)

        self.assertIn("compact", available_variant_names(replace(manifest, variants=(compact,))))
        self.assertEqual(result.variant_name, "compact")
        self.assertEqual(result.part_keys, tuple((part.side, part.layer_group_index) for part in manifest.sorted_parts()))
        np.testing.assert_allclose(result.kv, kv, atol=0.05)

    def test_partial_fetch_restores_requested_part_into_destination(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            self.write_artifact(tmp, source_key, kv)
            destination = np.full(kv.shape, np.nan, dtype=np.float32)
            result = LocalFetchDecodeController(tmp).fetch_decode(
                source_key,
                part_keys=(("k", 0),),
                destination=destination,
            )

        self.assertIs(result.kv, destination)
        self.assertEqual(result.part_keys, (("k", 0),))
        np.testing.assert_allclose(result.kv[0, 0:3, :, :, :], kv[0, 0:3, :, :, :], atol=0.05)
        self.assertTrue(np.isnan(result.kv[0, 3, :, :, :]).all())
        self.assertTrue(np.isnan(result.kv[1, :, :, :, :]).all())

    def test_partial_fetch_requires_destination(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            self.write_artifact(tmp, source_key, kv)
            with self.assertRaisesRegex(ValueError, "partial part fetch requires"):
                LocalFetchDecodeController(tmp).fetch_decode(source_key, part_keys=(("k", 0),))

    def test_fetch_decode_fails_on_corrupt_payload(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            store, _artifact_dir, _manifest = self.write_artifact(tmp, source_key, kv)
            (store.artifact_path(source_key) / "parts" / "k.g0.bin").write_bytes(b"corrupt")

            with self.assertRaisesRegex(ValueError, "payload size mismatch|checksum"):
                LocalFetchDecodeController(tmp).fetch_decode(source_key)


if __name__ == "__main__":
    unittest.main()
