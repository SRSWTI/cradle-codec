from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from cradle_codec.layout import FrameGeometry

from .api import EncodedPayload
from .raw_reference import checksum_bytes


_YUV444_NVDEC_MIN_DIM = 144

class PyNvVideoCodecUnavailableError(RuntimeError):
    pass


def _import_pynv() -> Any:
    try:
        import PyNvVideoCodec as nvc  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host CUDA/NVIDIA install
        raise PyNvVideoCodecUnavailableError(f"PyNvVideoCodec is unavailable: {exc}") from exc
    return nvc


def pynv_available() -> bool:
    try:
        _import_pynv()
    except PyNvVideoCodecUnavailableError:
        return False
    return True


def _validate_geometry(geometry: FrameGeometry) -> None:
    if geometry.channels != 3:
        raise ValueError(f"PyNvVideoCodec HEVC backend requires 3 channels, got {geometry.channels}")
    if geometry.encoded_height < geometry.logical_height or geometry.encoded_width < geometry.logical_width:
        raise ValueError(
            "encoded frame must cover logical frame: "
            f"encoded={(geometry.encoded_height, geometry.encoded_width)} "
            f"logical={(geometry.logical_height, geometry.logical_width)}"
        )


def _pad_to_encoded(frames: np.ndarray, geometry: FrameGeometry) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"expected frames rank 4 [T,H,W,3], got rank {frames.ndim}")
    if frames.dtype != np.uint8:
        raise TypeError(f"PyNvVideoCodec HEVC backend expects uint8 frames, got {frames.dtype}")
    if tuple(frames.shape[1:]) != geometry.logical_shape:
        raise ValueError(f"expected logical frame tail {geometry.logical_shape}, got {tuple(frames.shape[1:])}")
    if geometry.encoded_shape == geometry.logical_shape:
        return np.ascontiguousarray(frames)
    padded = np.zeros((frames.shape[0], *geometry.encoded_shape), dtype=np.uint8)
    padded[:, : geometry.logical_height, : geometry.logical_width, :] = frames
    return padded


def _frames_to_yuv444_planes(frames: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2))


def _feed_bytes(data: bytes) -> Callable[[bytearray], int]:
    view = memoryview(data)
    pos = 0

    def feed(dst: bytearray) -> int:
        nonlocal pos
        if pos >= len(view):
            return 0
        count = min(len(dst), len(view) - pos)
        dst[:count] = view[pos : pos + count]
        pos += count
        return count

    return feed


def _frame_to_numpy(frame: Any) -> np.ndarray:
    if hasattr(frame, "__dlpack__") and hasattr(np, "from_dlpack"):
        arr = np.from_dlpack(frame)
    else:
        arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        raise TypeError(f"decoded frame dtype is {arr.dtype}, expected uint8")
    if arr.ndim == 2 and arr.shape[0] % 3 == 0:
        plane_height = arr.shape[0] // 3
        arr = arr.reshape(3, plane_height, arr.shape[1]).transpose(1, 2, 0)
    if arr.ndim not in (2, 3):
        raise ValueError(f"decoded frame has unsupported shape {arr.shape}")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class PyNvVideoCodecHEVCCodec:
    """NVENC/NVDEC HEVC frame codec backed by NVIDIA PyNvVideoCodec.

    The performance backend is GPU-only: encode uses NVENC and decode uses
    NVDEC.  There is deliberately no implicit FFmpeg decode fallback here; use
    ``FFmpegHEVCCodec`` as an explicit backend when CPU decode is wanted.
    """

    gpu_id: int = 0
    preset: str = "P1"
    tuning_info: str = "lossless"
    rc: str = "constqp"
    constqp: int = 0
    fps: int = 30
    gop: int = 30
    bf: int = 0
    decode_with_nvdec: bool = True
    nvenc_workers: int = 1
    nvdec_workers: int = 1

    codec_name = "pynvvideocodec_hevc"

    def __post_init__(self) -> None:
        for name in ("nvenc_workers", "nvdec_workers"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
            object.__setattr__(self, name, value)

    def encoder_caps(self) -> dict[str, int]:
        nvc = _import_pynv()
        try:
            caps = nvc.GetEncoderCaps(self.gpu_id, "hevc")
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise PyNvVideoCodecUnavailableError(f"failed to query NVENC HEVC capabilities: {exc}") from exc
        if not caps.get("support_yuv444_encode", 0):
            raise PyNvVideoCodecUnavailableError("NVENC HEVC YUV444 encode is not supported on this GPU")
        if not caps.get("support_lossless_encode", 0):
            raise PyNvVideoCodecUnavailableError("NVENC HEVC lossless encode is not supported on this GPU")
        return {str(key): int(value) for key, value in caps.items() if isinstance(value, int)}

    def decoder_caps(self) -> dict[str, int]:
        nvc = _import_pynv()
        get_caps = getattr(nvc, "GetDecoderCaps", None)
        if not callable(get_caps):  # pragma: no cover - old PyNvVideoCodec
            return {}
        codec_enum = getattr(getattr(nvc, "cudaVideoCodec", None), "HEVC", None)
        chroma_enum = getattr(getattr(nvc, "cudaVideoChromaFormat", None), "444", None)
        if codec_enum is None or chroma_enum is None:  # pragma: no cover - API shape guard
            return {}
        try:
            caps = get_caps(self.gpu_id, codec_enum, chroma_enum, 8)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise PyNvVideoCodecUnavailableError(f"failed to query NVDEC HEVC YUV444 capabilities: {exc}") from exc
        if not caps.get("supported", 0):
            raise PyNvVideoCodecUnavailableError("NVDEC HEVC YUV444 decode is not supported on this GPU")
        return {str(key): int(value) for key, value in caps.items() if isinstance(value, int)}

    def payload_geometry(self, geometry: FrameGeometry) -> FrameGeometry:
        _validate_geometry(geometry)
        encoder_caps = self.encoder_caps()
        decoder_caps = self.decoder_caps()
        encoded_width = max(
            geometry.encoded_width,
            int(encoder_caps.get("width_min", 0)),
            int(decoder_caps.get("width_min", 0)),
            _YUV444_NVDEC_MIN_DIM,
        )
        encoded_height = max(
            geometry.encoded_height,
            int(encoder_caps.get("height_min", 0)),
            int(decoder_caps.get("height_min", 0)),
            _YUV444_NVDEC_MIN_DIM,
        )
        if encoded_width % 2:
            encoded_width += 1
        if encoded_height % 2:
            encoded_height += 1
        width_max_values = [int(value) for value in (encoder_caps.get("width_max", 0), decoder_caps.get("width_max", 0)) if int(value) > 0]
        height_max_values = [int(value) for value in (encoder_caps.get("height_max", 0), decoder_caps.get("height_max", 0)) if int(value) > 0]
        width_max = min(width_max_values) if width_max_values else encoded_width
        height_max = min(height_max_values) if height_max_values else encoded_height
        if encoded_width > width_max or encoded_height > height_max:
            raise ValueError(
                "encoded frame exceeds NVENC HEVC limits: "
                f"encoded={(encoded_height, encoded_width)} max={(height_max, width_max)}"
            )
        return FrameGeometry(
            logical_height=geometry.logical_height,
            logical_width=geometry.logical_width,
            encoded_height=encoded_height,
            encoded_width=encoded_width,
            channels=geometry.channels,
        )

    def _encoder_kwargs(self) -> dict[str, object]:
        return {
            "gpu_id": self.gpu_id,
            "codec": "hevc",
            "preset": self.preset,
            "tuning_info": self.tuning_info,
            "rc": self.rc,
            "constqp": self.constqp,
            "fps": self.fps,
            "gop": self.gop,
            "bf": self.bf,
        }

    def encode(self, frames: np.ndarray, geometry: FrameGeometry) -> EncodedPayload:
        _validate_geometry(geometry)
        nvc = _import_pynv()
        padded = _pad_to_encoded(frames, geometry)
        planes = _frames_to_yuv444_planes(padded)
        try:
            encoder = nvc.CreateEncoder(
                geometry.encoded_width,
                geometry.encoded_height,
                "YUV444",
                True,
                **self._encoder_kwargs(),
            )
            chunks: list[bytes] = []
            for frame in planes:
                packet = bytes(encoder.Encode(np.ascontiguousarray(frame)))
                if packet:
                    chunks.append(packet)
            tail = bytes(encoder.EndEncode())
            if tail:
                chunks.append(tail)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise RuntimeError(f"PyNvVideoCodec NVENC HEVC encode failed: {exc}") from exc
        data = b"".join(chunks)
        if not data:
            raise RuntimeError("PyNvVideoCodec NVENC HEVC encode produced an empty payload")
        return EncodedPayload(codec_name=self.codec_name, data=data, payload_bytes=len(data), checksum=checksum_bytes(data))

    def encode_many(self, requests: Iterable[tuple[np.ndarray, FrameGeometry]]) -> tuple[EncodedPayload, ...]:
        """Encode independent frame batches through a bounded NVENC worker pool."""

        items = tuple(requests)
        if len(items) <= 1 or self.nvenc_workers == 1:
            return tuple(self.encode(frames, geometry) for frames, geometry in items)
        with ThreadPoolExecutor(max_workers=self.nvenc_workers, thread_name_prefix="pynv-nvenc") as executor:
            return tuple(executor.map(lambda item: self.encode(item[0], item[1]), items))

    def _decode_with_pynv(self, data: bytes, geometry: FrameGeometry) -> np.ndarray:
        if not self.decode_with_nvdec:
            raise RuntimeError("PyNvVideoCodec HEVC decode requires NVDEC; choose the explicit ffmpeg backend for CPU decode")
        nvc = _import_pynv()
        demuxer = nvc.CreateDemuxer(_feed_bytes(data))
        decoder_kwargs: dict[str, object] = {}
        latency_type = getattr(nvc, "DisplayDecodeLatencyType", None)
        if latency_type is not None and hasattr(latency_type, "ZERO"):
            decoder_kwargs["latency"] = latency_type.ZERO
        decoder = nvc.CreateDecoder(
            gpuid=self.gpu_id,
            codec=demuxer.GetNvCodecId(),
            usedevicememory=False,
            maxwidth=geometry.encoded_width,
            maxheight=geometry.encoded_height,
            outputColorType=nvc.OutputColorType.NATIVE,
            **decoder_kwargs,
        )
        decoded_frames: list[np.ndarray] = []
        for packet in demuxer:
            decoded_frames.extend(_frame_to_numpy(frame) for frame in decoder.Decode(packet))
        flush = getattr(decoder, "Flush", None)
        if callable(flush):
            decoded_frames.extend(_frame_to_numpy(frame) for frame in flush())
        if not decoded_frames:
            raise RuntimeError("PyNvVideoCodec NVDEC returned no frames")
        decoded = np.stack(decoded_frames, axis=0)
        if decoded.shape[1:] == geometry.encoded_shape:
            return np.ascontiguousarray(decoded[:, : geometry.logical_height, : geometry.logical_width, :])
        if decoded.shape[1:] == (3, geometry.encoded_height, geometry.encoded_width):
            restored = decoded.transpose(0, 2, 3, 1)
            return np.ascontiguousarray(restored[:, : geometry.logical_height, : geometry.logical_width, :])
        raise ValueError(f"decoded NVDEC frames have unsupported shape {decoded.shape}")

    def decode(self, payload: EncodedPayload | bytes, geometry: FrameGeometry) -> np.ndarray:
        _validate_geometry(geometry)
        data = payload.data if isinstance(payload, EncodedPayload) else payload
        return self._decode_with_pynv(data, geometry)

    def decode_many(self, requests: Iterable[tuple[EncodedPayload | bytes, FrameGeometry]]) -> tuple[np.ndarray, ...]:
        """Decode independent HEVC payloads through a bounded NVDEC worker pool."""

        items = tuple(requests)
        if len(items) <= 1 or self.nvdec_workers == 1:
            return tuple(self.decode(payload, geometry) for payload, geometry in items)
        self._set_nvdec_session_count(self.nvdec_workers)
        with ThreadPoolExecutor(max_workers=self.nvdec_workers, thread_name_prefix="pynv-nvdec") as executor:
            return tuple(executor.map(lambda item: self.decode(item[0], item[1]), items))

    @staticmethod
    def _set_nvdec_session_count(worker_count: int) -> None:
        try:
            nvc = _import_pynv()
            decoder_class = getattr(nvc, "PyNvDecoder", None)
            set_session_count = getattr(decoder_class, "SetSessionCount", None)
            if callable(set_session_count):
                set_session_count(worker_count)
        except Exception:
            return
