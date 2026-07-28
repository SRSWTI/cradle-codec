import unittest

import numpy as np

from cradle_codec.codec import RawReferenceCodec, checksum_bytes
from cradle_codec.layout import FrameGeometry


class RawReferenceCodecTests(unittest.TestCase):
    def test_raw_reference_roundtrip_preserves_frames(self) -> None:
        frames = np.arange(4 * 3 * 5 * 3, dtype=np.uint16).reshape(4, 3, 5, 3)
        geometry = FrameGeometry(logical_height=3, logical_width=5, encoded_height=3, encoded_width=5)
        codec = RawReferenceCodec()

        payload = codec.encode(frames, geometry)
        decoded = codec.decode(payload, geometry)

        self.assertEqual(payload.codec_name, "raw_reference")
        self.assertEqual(payload.payload_bytes, len(payload.data))
        self.assertEqual(payload.checksum, checksum_bytes(payload.data))
        self.assertTrue(np.array_equal(decoded, frames))

    def test_raw_reference_rejects_wrong_geometry(self) -> None:
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        codec = RawReferenceCodec()
        payload = codec.encode(frames, FrameGeometry(4, 4, 4, 4))

        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            codec.decode(payload, FrameGeometry(2, 8, 2, 8))

    def test_raw_reference_corrupt_payload_fails_clearly(self) -> None:
        codec = RawReferenceCodec()

        with self.assertRaisesRegex(ValueError, "failed to decode"):
            codec.decode(b"not an npz", FrameGeometry(4, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
