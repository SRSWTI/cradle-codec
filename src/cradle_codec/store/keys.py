from __future__ import annotations

import hashlib
import re

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.@+-]+")


def sanitize_source_key(source_key: str, *, max_prefix: int = 96) -> str:
    cleaned = _SAFE_CHARS.sub("-", source_key).strip(".-")
    return (cleaned or "source")[:max_prefix]


def artifact_dir_name(source_key: str) -> str:
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
    return f"{sanitize_source_key(source_key)}--{digest}.kvcodec"
