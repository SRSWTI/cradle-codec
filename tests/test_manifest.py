import json
import tempfile
import unittest
from pathlib import Path

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
    read_manifest,
    verify_part_payload,
    write_manifest,
)
from cradle_codec.store import LocalArtifactStore, artifact_dir_name


class ManifestTests(unittest.TestCase):
    def _manifest(self) -> ArtifactManifest:
        quant = PartQuantizationManifest(
            mode="uint8_minmax",
            axis="channel",
            min_values=[0.0],
            scales=[1.0],
            source_dtype="float32",
            transport_dtype="uint8",
        )
        return ArtifactManifest(
            version=SCHEMA_VERSION,
            source_key="Qwen/Qwen3-8B@chunk/0",
            model="Qwen/Qwen3-8B",
            kv_shape=KVShapeManifest(2, 4, 8, 2, 3),
            layout=LayoutManifest(3, 1, 2, 1, 3),
            quantization=QuantizationManifest("uint8_minmax", "channel"),
            codec=CodecManifest("reference", "raw_reference", True, {}),
            parts=(
                ArtifactPart("v", 0, (0, 1, 2), 0, 8, 1, 6, 1, 6, "parts/v.g0.bin", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", quant),
                ArtifactPart("k", 0, (0, 1, 2), 0, 8, 1, 6, 1, 6, "parts/k.g0.bin", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", quant),
            ),
        )

    def test_manifest_roundtrips_json_with_stable_part_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, self._manifest())
            loaded = read_manifest(path)

        self.assertEqual([part.side for part in loaded.parts], ["k", "v"])
        self.assertEqual(loaded.kv_shape.num_layers, 4)
        self.assertEqual(loaded.layout.head_cols, 2)


    def test_manifest_accepts_legacy_single_variant_without_variants_key(self) -> None:
        manifest = self._manifest()
        data = {
            "version": manifest.version,
            "source_key": manifest.source_key,
            "model": manifest.model,
            "kv_shape": {
                "num_sides": manifest.kv_shape.num_sides,
                "num_layers": manifest.kv_shape.num_layers,
                "num_tokens": manifest.kv_shape.num_tokens,
                "num_kv_heads": manifest.kv_shape.num_kv_heads,
                "head_dim": manifest.kv_shape.head_dim,
            },
            "layout": {
                "layers_per_frame": manifest.layout.layers_per_frame,
                "head_rows": manifest.layout.head_rows,
                "head_cols": manifest.layout.head_cols,
                "dim_rows": manifest.layout.dim_rows,
                "dim_cols": manifest.layout.dim_cols,
            },
            "quantization": {"mode": manifest.quantization.mode, "axis": manifest.quantization.axis},
            "codec": {
                "family": manifest.codec.family,
                "backend": manifest.codec.backend,
                "lossless_video": manifest.codec.lossless_video,
                "codec_params": manifest.codec.codec_params,
            },
            "parts": json.loads(json.dumps([part.__dict__ | {"quantization": part.quantization.__dict__} for part in manifest.sorted_parts()])),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = read_manifest(path)

        self.assertEqual(loaded.variants, ())
        self.assertEqual(loaded.base_variant().payload_bytes, 6)
        self.assertEqual([part.side for part in loaded.sorted_parts()], ["k", "v"])

    def test_manifest_roundtrips_named_variants(self) -> None:
        manifest = self._manifest()
        variant = ArtifactVariant(
            name="low",
            parts=manifest.sorted_parts(),
            payload_bytes=sum(part.payload_bytes for part in manifest.parts),
            estimated_decode_ms=2.5,
        )
        manifest = ArtifactManifest(
            version=manifest.version,
            source_key=manifest.source_key,
            model=manifest.model,
            kv_shape=manifest.kv_shape,
            layout=manifest.layout,
            quantization=manifest.quantization,
            codec=manifest.codec,
            parts=manifest.parts,
            variants=(variant,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, manifest)
            loaded = read_manifest(path)

        self.assertEqual(loaded.variant("low").estimated_decode_ms, 2.5)
        self.assertEqual(loaded.variant("low").payload_bytes, 6)
        self.assertEqual([part.payload_path for part in loaded.sorted_parts("low")], ["parts/k.g0.bin", "parts/v.g0.bin"])

    def test_verify_part_payload_catches_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parts").mkdir()
            (root / "parts" / "k.g0.bin").write_bytes(b"abc")
            part = self._manifest().sorted_parts()[0]
            self.assertEqual(verify_part_payload(root, part), b"abc")
            (root / "parts" / "k.g0.bin").write_bytes(b"abd")

            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_part_payload(root, part)

    def test_local_store_detects_complete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            artifact = store.artifact_path("Qwen/Qwen3-8B@chunk/0")
            (artifact / "parts").mkdir(parents=True)
            (artifact / "parts" / "k.g0.bin").write_bytes(b"abc")
            (artifact / "parts" / "v.g0.bin").write_bytes(b"abc")
            write_manifest(artifact / "manifest.json", self._manifest())

            self.assertTrue(store.has_complete_artifact("Qwen/Qwen3-8B@chunk/0"))
            (artifact / "parts" / "v.g0.bin").unlink()
            self.assertFalse(store.has_complete_artifact("Qwen/Qwen3-8B@chunk/0"))

    def test_artifact_dir_name_is_stable_and_safe(self) -> None:
        name = artifact_dir_name("Qwen/Qwen3-8B@chunk/0")

        self.assertIn("Qwen-Qwen3-8B@chunk-0", name)
        self.assertTrue(name.endswith(".kvcodec"))


if __name__ == "__main__":
    unittest.main()
