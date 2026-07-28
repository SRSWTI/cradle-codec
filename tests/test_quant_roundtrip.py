import unittest

import numpy as np

from cradle_codec.quant import (
    compute_error_metrics,
    dequantize_uint8_minmax,
    quantize_uint8_minmax,
)


class QuantizationRoundtripTests(unittest.TestCase):

    def test_uint8_minmax_channel_roundtrip_has_bounded_error(self) -> None:
        values = np.linspace(-2.0, 5.0, num=4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)

        quantized, metadata = quantize_uint8_minmax(values, axis="channel")
        reconstructed = dequantize_uint8_minmax(quantized, metadata)
        metrics = compute_error_metrics(values, reconstructed)

        self.assertEqual(quantized.dtype, np.uint8)
        self.assertEqual(metadata.mode, "uint8_minmax")
        self.assertEqual(metadata.axis, "channel")
        self.assertLessEqual(metrics.max_abs_error, float(np.max(metadata.scales)) / 2.0 + 1e-6)
        self.assertGreater(metrics.cosine_similarity or 0.0, 0.9999)

    def test_uint8_minmax_frame_metadata_reconstructs_shape(self) -> None:
        values = np.stack(
            [
                np.full((2, 3, 2), -1.0, dtype=np.float32),
                np.full((2, 3, 2), 7.0, dtype=np.float32),
            ],
            axis=0,
        )

        quantized, metadata = quantize_uint8_minmax(values, axis="frame")
        reconstructed = dequantize_uint8_minmax(quantized, metadata)

        self.assertEqual(metadata.min_values.shape, (2,))
        self.assertEqual(metadata.scales.shape, (2,))
        self.assertTrue(np.array_equal(reconstructed, values))

    def test_uint8_minmax_general_values_have_quantization_error(self) -> None:
        values = np.array([0.0, 0.1, 0.2, 0.3333, 1.0], dtype=np.float32)

        quantized, metadata = quantize_uint8_minmax(values, axis="part")
        reconstructed = dequantize_uint8_minmax(quantized, metadata)
        metrics = compute_error_metrics(values, reconstructed)

        self.assertGreater(metrics.max_abs_error, 0.0)
        self.assertLess(metrics.max_abs_error, 1.0 / 255.0)

    def test_error_metrics_report_expected_fields(self) -> None:
        reference = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        reconstructed = np.array([0.0, 1.5, 1.5], dtype=np.float32)

        metrics = compute_error_metrics(reference, reconstructed)

        self.assertEqual(metrics.max_abs_error, 0.5)
        self.assertAlmostEqual(metrics.mean_abs_error, 1.0 / 3.0)
        self.assertGreater(metrics.rmse, 0.0)
        self.assertIn("cosine_similarity", metrics.as_dict())


if __name__ == "__main__":
    unittest.main()
