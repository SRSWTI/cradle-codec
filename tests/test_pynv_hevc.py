from __future__ import annotations

import unittest

import numpy as np

from cradle_codec.codec import EncodedPayload, PyNvVideoCodecHEVCCodec, checksum_bytes
from cradle_codec.layout import FrameGeometry


class _CountingPyNvCodec(PyNvVideoCodecHEVCCodec):
    """Test double that exercises PyNv worker-pool plumbing without NVENC/NVDEC."""

    def __init__(self, *, nvenc_workers: int = 1, nvdec_workers: int = 1) -> None:
        super().__init__(nvenc_workers=nvenc_workers, nvdec_workers=nvdec_workers)
        object.__setattr__(self, "encoded_values", [])
        object.__setattr__(self, "decoded_values", [])

    def encode(self, frames: np.ndarray, geometry: FrameGeometry) -> EncodedPayload:
        value = int(frames[0, 0, 0, 0])
        self.encoded_values.append(value)
        data = bytes([value])
        return EncodedPayload(codec_name=self.codec_name, data=data, payload_bytes=len(data), checksum=checksum_bytes(data))

    def decode(self, payload: EncodedPayload | bytes, geometry: FrameGeometry) -> np.ndarray:
        data = payload.data if isinstance(payload, EncodedPayload) else payload
        value = int(data[0])
        self.decoded_values.append(value)
        return np.full((1, geometry.logical_height, geometry.logical_width, geometry.channels), value, dtype=np.uint8)

    @staticmethod
    def _set_nvdec_session_count(worker_count: int) -> None:
        return None



class _CapsPyNvCodec(PyNvVideoCodecHEVCCodec):
    def encoder_caps(self) -> dict[str, int]:
        return {"width_min": 130, "height_min": 34, "width_max": 4096, "height_max": 4096}

    def decoder_caps(self) -> dict[str, int]:
        return {"width_min": 144, "height_min": 144, "width_max": 4096, "height_max": 4096}

class PyNvHEVCCodecTests(unittest.TestCase):
    def test_worker_counts_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "nvenc_workers"):
            PyNvVideoCodecHEVCCodec(nvenc_workers=0)
        with self.assertRaisesRegex(ValueError, "nvdec_workers"):
            PyNvVideoCodecHEVCCodec(nvdec_workers=0)

    def test_payload_geometry_satisfies_nvdec_minimums(self) -> None:
        codec = _CapsPyNvCodec()
        geometry = codec.payload_geometry(FrameGeometry(logical_height=1, logical_width=1024, encoded_height=33, encoded_width=1024))

        self.assertEqual(geometry.logical_height, 1)
        self.assertEqual(geometry.logical_width, 1024)
        self.assertEqual(geometry.encoded_height, 144)
        self.assertEqual(geometry.encoded_width, 1024)

    def test_decode_requires_nvdec_without_implicit_ffmpeg_fallback(self) -> None:
        codec = PyNvVideoCodecHEVCCodec(decode_with_nvdec=False)
        logical = FrameGeometry(logical_height=32, logical_width=32, encoded_height=32, encoded_width=32)

        with self.assertRaisesRegex(RuntimeError, "requires NVDEC"):
            codec.decode(b"not-a-hevc-stream", logical)

    def test_encode_many_and_decode_many_preserve_request_order(self) -> None:
        codec = _CountingPyNvCodec(nvenc_workers=2, nvdec_workers=2)
        geometry = FrameGeometry(logical_height=4, logical_width=4, encoded_height=4, encoded_width=4)
        requests = tuple((np.full((1, 4, 4, 3), value, dtype=np.uint8), geometry) for value in (3, 7, 11, 13))

        payloads = codec.encode_many(requests)
        decoded = codec.decode_many((payload, geometry) for payload in payloads)

        self.assertEqual([payload.data[0] for payload in payloads], [3, 7, 11, 13])
        self.assertEqual([int(frames[0, 0, 0, 0]) for frames in decoded], [3, 7, 11, 13])
        self.assertCountEqual(codec.encoded_values, [3, 7, 11, 13])
        self.assertCountEqual(codec.decoded_values, [3, 7, 11, 13])

    def test_pynv_hevc_roundtrip_preserves_uint8_frames_with_nvdec(self) -> None:
        codec = PyNvVideoCodecHEVCCodec()
        logical = FrameGeometry(logical_height=32, logical_width=32, encoded_height=32, encoded_width=32)
        try:
            geometry = codec.payload_geometry(logical)
        except Exception as exc:
            self.skipTest(f"PyNvVideoCodec NVENC unavailable: {exc}")

        frames = ((np.arange(4 * 32 * 32 * 3, dtype=np.uint32) * 17 + 3) % 256).astype(np.uint8).reshape(4, 32, 32, 3)
        try:
            payload = codec.encode(frames, geometry)
            decoded = codec.decode(payload, geometry)
        except Exception as exc:
            self.skipTest(f"PyNvVideoCodec HEVC unavailable on this host: {exc}")

        self.assertEqual(payload.codec_name, "pynvvideocodec_hevc")
        self.assertGreater(payload.payload_bytes, 0)
        self.assertEqual(decoded.shape, frames.shape)
        self.assertTrue(np.array_equal(decoded, frames))


if __name__ == "__main__":
    unittest.main()
