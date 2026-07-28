import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from cradle_codec.cli import app
from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.pipeline import encode_kv_chunk
from cradle_codec.remote import serve_artifacts
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

class RemoteCliTests(unittest.TestCase):
    def run_server(self, root: str):
        server = serve_artifacts(root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(server.shutdown)
        host, port = server.server_address
        return f"http://{host}:{port}"

    def test_remote_fetch_command_restores_artifact(self) -> None:
        runner = CliRunner()
        source_key = "synthetic/chunk0"
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            encode_kv_chunk(kv, store.artifact_path(source_key), source_key=source_key, model="synthetic", layout=layout())
            base_url = self.run_server(tmp)
            output = Path(tmp) / "restored.npy"

            result = runner.invoke(
                app,
                [
                    "remote",
                    "fetch",
                    "--base-url",
                    base_url,
                    "--source-key",
                    source_key,
                    "--output",
                    str(output),
                    "--codec",
                    "reference",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("variant=base", result.output)
            restored = np.load(output)
            self.assertEqual(restored.shape, kv.shape)
            self.assertLess(float(np.max(np.abs(restored - kv))), 0.05)


if __name__ == "__main__":
    unittest.main()
