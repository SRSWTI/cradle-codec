from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from cradle_codec.codec import checksum_bytes

from .schema import ArtifactManifest, ArtifactPart, manifest_from_dict


def artifact_payload_path(artifact_dir: str | Path, payload_path: str) -> Path:
    """Resolve a manifest payload path without allowing absolute/traversal paths."""

    if "\\" in payload_path:
        raise ValueError(f"artifact payload path must use forward slashes: {payload_path!r}")
    relative = PurePosixPath(payload_path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"artifact payload path must be a safe relative path: {payload_path!r}")
    return Path(artifact_dir).joinpath(*relative.parts)


def read_manifest(path: str | Path) -> ArtifactManifest:
    path = Path(path)
    return manifest_from_dict(json.loads(path.read_text(encoding="utf-8")))


def verify_part_payload_bytes(data: bytes, part: ArtifactPart) -> bytes:
    if len(data) != part.payload_bytes:
        raise ValueError(f"payload size mismatch for {part.payload_path}: {len(data)} != {part.payload_bytes}")
    actual = checksum_bytes(data)
    if actual != part.checksum:
        raise ValueError(f"payload checksum mismatch for {part.payload_path}: {actual} != {part.checksum}")
    return data


def verify_part_payload(artifact_dir: str | Path, part: ArtifactPart) -> bytes:
    payload_path = artifact_payload_path(artifact_dir, part.payload_path)
    if not payload_path.is_file():
        raise FileNotFoundError(f"missing artifact part: {part.payload_path}")
    return verify_part_payload_bytes(payload_path.read_bytes(), part)
