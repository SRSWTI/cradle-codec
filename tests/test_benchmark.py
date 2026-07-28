import tempfile
import unittest

import numpy as np

from cradle_codec.benchmark import benchmark_artifact_reuse, benchmark_report_to_dict
from cradle_codec.layout import HeadDimTiling, KVCodecLayout
from cradle_codec.pipeline import encode_kv_chunk


def synthetic_kv() -> np.ndarray:
    values = np.arange(2 * 4 * 5 * 2 * 3, dtype=np.float32).reshape(2, 4, 5, 2, 3)
    return (values / 17.0 - 3.0).astype(np.float32)


class BenchmarkTests(unittest.TestCase):
    def layout(self) -> KVCodecLayout:
        return KVCodecLayout(
            num_layers=4,
            num_kv_heads=2,
            head_dim=3,
            layers_per_frame=3,
            tiling=HeadDimTiling(head_rows=1, head_cols=2, dim_rows=1, dim_cols=3),
        )

    def test_benchmark_reports_transfer_decode_restore_and_accuracy(self) -> None:
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = encode_kv_chunk(kv, tmp, source_key="synthetic/chunk0", model="synthetic", layout=self.layout())

            report = benchmark_artifact_reuse(
                tmp,
                expected_kv=kv,
                bandwidth_bytes_per_sec=1024.0,
                prefill_ms=250.0,
                scheduler_wait_ms=7.0,
            )

        metrics = {metric.method: metric for metric in report.metrics}
        self.assertEqual(report.source_key, "synthetic/chunk0")
        self.assertEqual(metrics["raw_kv_reuse"].network_bytes, kv.nbytes)
        self.assertEqual(metrics["codec_reuse"].network_bytes, sum(part.payload_bytes for part in manifest.parts))
        self.assertEqual(metrics["codec_reuse"].selected_variant, "base")
        self.assertLess(metrics["codec_reuse"].max_abs_error or 1.0, 0.05)
        self.assertGreater(metrics["codec_reuse"].decode_ms, 0.0)
        self.assertGreater(metrics["codec_reuse"].restore_ms, 0.0)

    def test_report_serializes_as_plain_json(self) -> None:
        kv = synthetic_kv()
        with tempfile.TemporaryDirectory() as tmp:
            encode_kv_chunk(kv, tmp, source_key="synthetic/chunk0", model="synthetic", layout=self.layout())
            report = benchmark_artifact_reuse(tmp, expected_kv=kv, bandwidth_bytes_per_sec=4096.0, prefill_ms=100.0)

        data = benchmark_report_to_dict(report)

        self.assertEqual(data["model"], "synthetic")
        self.assertIn("metrics", data)
        self.assertEqual({metric["method"] for metric in data["metrics"]}, {"full_prefill", "raw_kv_reuse", "codec_reuse"})


if __name__ == "__main__":
    unittest.main()
