from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from cradle_codec.layout import FrameGeometry

from .api import EncodedPayload
from .raw_reference import checksum_bytes


class FFmpegUnavailableError(RuntimeError):
    pass


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _validate_geometry(geometry: FrameGeometry) -> None:
    if geometry.channels != 3:
        raise ValueError(f"HEVC YUV444 backend requires 3 channels, got {geometry.channels}")
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
        raise TypeError(f"HEVC backend expects uint8 frames, got {frames.dtype}")
    if tuple(frames.shape[1:]) != geometry.logical_shape:
        raise ValueError(f"expected logical frame tail {geometry.logical_shape}, got {tuple(frames.shape[1:])}")
    if geometry.encoded_shape == geometry.logical_shape:
        return np.ascontiguousarray(frames)
    padded = np.zeros((frames.shape[0], *geometry.encoded_shape), dtype=np.uint8)
    padded[:, : geometry.logical_height, : geometry.logical_width, :] = frames
    return padded


def _frames_to_yuv444p_bytes(frames: np.ndarray) -> bytes:
    # FFmpeg raw yuv444p expects each frame as full Y plane, then U plane, then V plane.
    planar = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    return planar.tobytes()


def _yuv444p_bytes_to_frames(raw: bytes, *, height: int, width: int) -> np.ndarray:
    frame_bytes = height * width * 3
    if frame_bytes <= 0:
        raise ValueError("invalid encoded frame dimensions")
    if len(raw) == 0 or len(raw) % frame_bytes != 0:
        raise ValueError(f"decoded rawvideo size {len(raw)} is not a multiple of one frame ({frame_bytes})")
    frame_count = len(raw) // frame_bytes
    planar = np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, 3, height, width)
    return np.ascontiguousarray(planar.transpose(0, 2, 3, 1))


@dataclass(frozen=True)
class FFmpegHEVCCodec:
    """HEVC frame codec using FFmpeg raw YUV444 pipes.

    Frames are treated as integer Y/U/V planes. The configured encoder must use a
    lossless profile or an equivalent no-loss mode; this class validates the frame
    container and dimensions, while round-trip tests validate preservation on the
    local FFmpeg build.
    """

    ffmpeg_path: str | None = None
    encoder: str = "libx265"
    preset: str = "ultrafast"
    extra_encoder_args: tuple[str, ...] = ()

    codec_name = "ffmpeg_hevc"

    def _ffmpeg(self) -> str:
        path = self.ffmpeg_path or find_ffmpeg()
        if path is None:
            raise FFmpegUnavailableError("ffmpeg executable not found")
        return path

    def _encoder_args(self) -> list[str]:
        if self.extra_encoder_args:
            return list(self.extra_encoder_args)
        if self.encoder == "libx265":
            return ["-c:v", "libx265", "-preset", self.preset, "-x265-params", "lossless=1", "-pix_fmt", "yuv444p"]
        if self.encoder == "hevc_nvenc":
            return ["-c:v", "hevc_nvenc", "-preset", "p1", "-tune", "lossless", "-pix_fmt", "yuv444p"]
        return ["-c:v", self.encoder, "-pix_fmt", "yuv444p"]

    def encode(self, frames: np.ndarray, geometry: FrameGeometry) -> EncodedPayload:
        _validate_geometry(geometry)
        padded = _pad_to_encoded(frames, geometry)
        height = geometry.encoded_height
        width = geometry.encoded_width
        cmd = [
            self._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p",
            "-s:v",
            f"{width}x{height}",
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-frames:v",
            str(padded.shape[0]),
            "-an",
            *self._encoder_args(),
            "-f",
            "hevc",
            "pipe:1",
        ]
        proc = subprocess.run(
            cmd,
            input=_frames_to_yuv444p_bytes(padded),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg HEVC encode failed with code {proc.returncode}: {stderr}")
        data = proc.stdout
        if not data:
            raise RuntimeError("ffmpeg HEVC encode produced an empty payload")
        return EncodedPayload(codec_name=self.codec_name, data=data, payload_bytes=len(data), checksum=checksum_bytes(data))

    def decode(self, payload: EncodedPayload | bytes, geometry: FrameGeometry) -> np.ndarray:
        _validate_geometry(geometry)
        data = payload.data if isinstance(payload, EncodedPayload) else payload
        cmd = [
            self._ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "hevc",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg HEVC decode failed with code {proc.returncode}: {stderr}")
        decoded = _yuv444p_bytes_to_frames(proc.stdout, height=geometry.encoded_height, width=geometry.encoded_width)
        return np.ascontiguousarray(decoded[:, : geometry.logical_height, : geometry.logical_width, :])
