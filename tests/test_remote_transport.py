import tempfile
import threading
import unittest

import numpy as np

from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.pipeline import encode_kv_chunk
from cradle_codec.remote import ArtifactHttpClient, ArtifactHttpError, RemoteFetchDecodeController, serve_artifacts
from cradle_codec.store import LocalArtifactStore


def synthetic_kv() -> np.ndarray:
    values = np.arange(2 * 4 * 5 * 2 * 3, dtype=np.float32).reshape(2, 4, 5, 2, 3)
    return (values / 19.0 - 2.0).astype(np.float32)


def layout() -> KVCodecLayout:
    return KVCodecLayout(
        num_layers=4,
        num_kv_heads=2,
        head_dim=3,
        layers_per_frame=3,
        tiling=HeadDimTiling(head_rows=1, head_cols=2, dim_rows=1, dim_cols=3),
    )


class RemoteTransportTests(unittest.TestCase):
    def run_server(self, root: str):
        server = serve_artifacts(root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(server.shutdown)
        host, port = server.server_address
        return f"http://{host}:{port}"

    def test_remote_fetch_decode_restores_artifact(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            artifact_dir = store.artifact_path(source_key)
            encode_kv_chunk(kv, artifact_dir, source_key=source_key, model="synthetic", layout=layout())
            base_url = self.run_server(tmp)

            result = RemoteFetchDecodeController(ArtifactHttpClient(base_url)).fetch_decode(source_key)

        self.assertEqual(result.source_key, source_key)
        self.assertEqual(result.variant_name, None)
        self.assertGreater(result.timing.transfer_ms, 0.0)
        np.testing.assert_allclose(result.kv, kv, atol=0.05)

    def test_remote_fetch_rejects_corrupt_payload(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            artifact_dir = store.artifact_path(source_key)
            encode_kv_chunk(kv, artifact_dir, source_key=source_key, model="synthetic", layout=layout())
            (artifact_dir / "parts" / "k.g0.bin").write_bytes(b"corrupt")
            base_url = self.run_server(tmp)

            with self.assertRaisesRegex(ValueError, "payload size mismatch|checksum"):
                RemoteFetchDecodeController(base_url).fetch_decode(source_key)

    def test_server_rejects_unsafe_payload_path(self) -> None:
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            encode_kv_chunk(kv, store.artifact_path(source_key), source_key=source_key, model="synthetic", layout=layout())
            client = ArtifactHttpClient(self.run_server(tmp))

            with self.assertRaisesRegex(ArtifactHttpError, "400"):
                client.fetch_part(source_key, "../secret.bin")


if __name__ == "__main__":
    unittest.main()
