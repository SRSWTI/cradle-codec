from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from cradle_codec.manifest import ArtifactManifest
from cradle_codec.store import artifact_dir_name


JsonPrimitive = str | int | float | bool | None
JsonLike = JsonPrimitive | list["JsonLike"] | dict[str, "JsonLike"]


def _stable_jsonable(value: Any) -> JsonLike:
    if is_dataclass(value) and not isinstance(value, type):
        return _stable_jsonable(asdict(value))
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_stable_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    return str(value)


def normalize_lmcache_key(key: Any) -> str:
    """Return a deterministic, dependency-free string form for an LMCache key.

    The real LMCache key types are intentionally not imported here.  Strings are
    treated as already canonical; structured Python values are encoded as stable
    JSON so callers can use tuples/dicts in tests and adapters without pulling in
    LMCache itself.
    """

    if isinstance(key, str):
        if not key:
            raise ValueError("LMCache key must not be empty")
        return key
    normalized = json.dumps(_stable_jsonable(key), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if not normalized:
        raise ValueError("LMCache key must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class LMCacheArtifactKey:
    """Stable bridge between an LMCache cache key and a KVCodec artifact key."""

    model: str
    cache_key: str
    token_start: int = 0
    token_count: int | None = None
    namespace: str = "lmcache"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.cache_key:
            raise ValueError("cache_key must not be empty")
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count is not None and self.token_count <= 0:
            raise ValueError("token_count must be positive when provided")
        if not self.namespace:
            raise ValueError("namespace must not be empty")

    @classmethod
    def from_lmcache_key(
        cls,
        *,
        model: str,
        key: Any,
        token_start: int = 0,
        token_count: int | None = None,
        namespace: str = "lmcache",
    ) -> "LMCacheArtifactKey":
        return cls(
            model=model,
            cache_key=normalize_lmcache_key(key),
            token_start=token_start,
            token_count=token_count,
            namespace=namespace,
        )

    @property
    def token_span(self) -> str:
        count = "*" if self.token_count is None else str(self.token_count)
        return f"{self.token_start}+{count}"

    @property
    def source_key(self) -> str:
        return f"{self.namespace}:{self.model}@{self.cache_key}#tokens={self.token_span}"

    @property
    def artifact_dir_name(self) -> str:
        return artifact_dir_name(self.source_key)


def lmcache_source_key(
    *,
    model: str,
    key: Any,
    token_start: int = 0,
    token_count: int | None = None,
    namespace: str = "lmcache",
) -> str:
    return LMCacheArtifactKey.from_lmcache_key(
        model=model,
        key=key,
        token_start=token_start,
        token_count=token_count,
        namespace=namespace,
    ).source_key


def lmcache_artifact_dir_name(
    *,
    model: str,
    key: Any,
    token_start: int = 0,
    token_count: int | None = None,
    namespace: str = "lmcache",
) -> str:
    return LMCacheArtifactKey.from_lmcache_key(
        model=model,
        key=key,
        token_start=token_start,
        token_count=token_count,
        namespace=namespace,
    ).artifact_dir_name


def is_lmcache_compatible_manifest(manifest: ArtifactManifest, key: LMCacheArtifactKey) -> bool:
    if manifest.model != key.model:
        return False
    if manifest.source_key != key.source_key:
        return False
    if key.token_count is not None and manifest.kv_shape.num_tokens != key.token_count:
        return False
    return True


def require_lmcache_compatible_manifest(manifest: ArtifactManifest, key: LMCacheArtifactKey) -> ArtifactManifest:
    """Validate that a decoded artifact belongs to the expected LMCache key."""

    if manifest.model != key.model:
        raise ValueError(f"manifest model {manifest.model!r} does not match LMCache model {key.model!r}")
    if manifest.source_key != key.source_key:
        raise ValueError("manifest source_key does not match LMCache artifact key")
    if key.token_count is not None and manifest.kv_shape.num_tokens != key.token_count:
        raise ValueError(
            f"manifest token count {manifest.kv_shape.num_tokens} does not match LMCache span {key.token_count}"
        )
    return manifest
