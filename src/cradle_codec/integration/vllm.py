from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


QWEN3_17B_MODEL = "Qwen/Qwen3-1.7B"
KVRole = Literal["kv_producer", "kv_consumer", "kv_both"]
LMCacheMode = Literal["mp", "in_process"]

_VALID_KV_ROLES = {"kv_producer", "kv_consumer", "kv_both"}
_LMCACHE_MP_MODULE_PATH = "lmcache.integration.vllm.lmcache_mp_connector"
_KVCODEC_LMCACHE_PLUGIN_NAME = "cradle_codec"
_KVCODEC_LMCACHE_PLUGIN_MODULE = "cradle_codec.integration.lmcache_storage"
_KVCODEC_LMCACHE_PLUGIN_CLASS = "KVCodecLMCacheStorageBackend"
_VALID_KVCODEC_CODECS = {"reference", "ffmpeg", "pynv", "pynvvideocodec"}
_VALID_QUANTIZATION_AXES = {"part", "frame", "channel"}
CodecBackend = Literal["reference", "ffmpeg", "pynv", "pynvvideocodec"]
QuantizationAxis = Literal["part", "frame", "channel"]


def _positive_int(name: str, value: int | None) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return value


def _non_empty_optional(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty when provided")
    return normalized


def _path_string(path: str | Path) -> str:
    value = str(path)
    if not value:
        raise ValueError("artifact_root must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class KVCodecLMCacheStoragePluginConfig:
    """LMCache storage plugin configuration for the cradle_codec backend.

    LMCache 0.5.x enables Python storage backends with ``storage_plugins`` plus
    ``extra_config.storage_plugin.<name>.*`` keys. This object deliberately
    stores only serializable launch-time values so callers can emit config/env
    without importing LMCache or the backend implementation.
    """

    enabled: bool = True
    plugin_name: str = _KVCODEC_LMCACHE_PLUGIN_NAME
    module_path: str = _KVCODEC_LMCACHE_PLUGIN_MODULE
    class_name: str = _KVCODEC_LMCACHE_PLUGIN_CLASS
    artifact_root: str | Path = ".cradle-codec"
    codec: CodecBackend = "reference"
    quantization_axis: QuantizationAxis = "channel"
    layout_name: str | None = None
    layers_per_frame: int | None = None
    head_rows: int | None = None
    head_cols: int | None = None
    dim_rows: int | None = None
    dim_cols: int | None = None
    io_threads: int | None = None
    nvenc_workers: int | None = None
    nvdec_workers: int | None = None

    def __post_init__(self) -> None:
        plugin_name = self.plugin_name.strip()
        if not plugin_name or "," in plugin_name:
            raise ValueError("plugin_name must be non-empty and must not contain ','")
        module_path = self.module_path.strip()
        class_name = self.class_name.strip()
        if not module_path:
            raise ValueError("module_path must not be empty")
        if not class_name:
            raise ValueError("class_name must not be empty")
        codec = self.codec.strip()
        if codec not in _VALID_KVCODEC_CODECS:
            raise ValueError("codec must be one of: reference, ffmpeg, pynv, pynvvideocodec")
        quantization_axis = self.quantization_axis.strip()
        if quantization_axis not in _VALID_QUANTIZATION_AXES:
            raise ValueError("quantization_axis must be one of: part, frame, channel")
        tiling_values = (self.head_rows, self.head_cols, self.dim_rows, self.dim_cols)
        if any(value is not None for value in tiling_values) and not all(value is not None for value in tiling_values):
            raise ValueError("head_rows, head_cols, dim_rows, and dim_cols must be provided together")
        object.__setattr__(self, "plugin_name", plugin_name)
        object.__setattr__(self, "module_path", module_path)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "artifact_root", _path_string(self.artifact_root))
        object.__setattr__(self, "codec", codec)
        object.__setattr__(self, "quantization_axis", quantization_axis)
        object.__setattr__(self, "layout_name", _non_empty_optional("layout_name", self.layout_name))
        object.__setattr__(self, "layers_per_frame", _positive_int("layers_per_frame", self.layers_per_frame))
        object.__setattr__(self, "head_rows", _positive_int("head_rows", self.head_rows))
        object.__setattr__(self, "head_cols", _positive_int("head_cols", self.head_cols))
        object.__setattr__(self, "dim_rows", _positive_int("dim_rows", self.dim_rows))
        object.__setattr__(self, "dim_cols", _positive_int("dim_cols", self.dim_cols))
        object.__setattr__(self, "io_threads", _positive_int("io_threads", self.io_threads))
        object.__setattr__(self, "nvenc_workers", _positive_int("nvenc_workers", self.nvenc_workers))
        object.__setattr__(self, "nvdec_workers", _positive_int("nvdec_workers", self.nvdec_workers))

    def storage_plugins(self) -> tuple[str, ...]:
        return (self.plugin_name,) if self.enabled else ()

    def extra_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        prefix = f"storage_plugin.{self.plugin_name}"
        config: dict[str, Any] = {
            f"{prefix}.module_path": self.module_path,
            f"{prefix}.class_name": self.class_name,
            f"{prefix}.artifact_root": self.artifact_root,
            f"{prefix}.codec": self.codec,
            f"{prefix}.quantization_axis": self.quantization_axis,
        }
        optional_values = {
            "layout_name": self.layout_name,
            "layers_per_frame": self.layers_per_frame,
            "head_rows": self.head_rows,
            "head_cols": self.head_cols,
            "dim_rows": self.dim_rows,
            "dim_cols": self.dim_cols,
            "io_threads": self.io_threads,
            "nvenc_workers": self.nvenc_workers,
            "nvdec_workers": self.nvdec_workers,
        }
        config.update({f"{prefix}.{key}": value for key, value in optional_values.items() if value is not None})
        return config

    def lmcache_config(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "storage_plugins": list(self.storage_plugins()),
            "extra_config": self.extra_config(),
        }

    def env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(base_env or {})
        if not self.enabled:
            return env
        existing_plugins = [item.strip() for item in env.get("LMCACHE_STORAGE_PLUGINS", "").split(",") if item.strip()]
        if self.plugin_name not in existing_plugins:
            existing_plugins.append(self.plugin_name)
        existing_extra: dict[str, Any] = {}
        if raw_extra := env.get("LMCACHE_EXTRA_CONFIG"):
            try:
                parsed = json.loads(raw_extra)
            except json.JSONDecodeError as exc:
                raise ValueError("base_env LMCACHE_EXTRA_CONFIG must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("base_env LMCACHE_EXTRA_CONFIG must decode to a JSON object")
            existing_extra.update(parsed)
        existing_extra.update(self.extra_config())
        env["LMCACHE_STORAGE_PLUGINS"] = ",".join(existing_plugins)
        env["LMCACHE_EXTRA_CONFIG"] = json.dumps(existing_extra, sort_keys=True, separators=(",", ":"))
        return env

def lmcache_kvcodec_storage_plugin_config(
    *,
    enabled: bool = True,
    plugin_name: str = _KVCODEC_LMCACHE_PLUGIN_NAME,
    artifact_root: str | Path = ".cradle-codec",
    codec: CodecBackend = "reference",
    quantization_axis: QuantizationAxis = "channel",
    layout_name: str | None = None,
    layers_per_frame: int | None = None,
    head_rows: int | None = None,
    head_cols: int | None = None,
    dim_rows: int | None = None,
    dim_cols: int | None = None,
    io_threads: int | None = None,
    nvenc_workers: int | None = None,
    nvdec_workers: int | None = None,
    module_path: str = _KVCODEC_LMCACHE_PLUGIN_MODULE,
    class_name: str = _KVCODEC_LMCACHE_PLUGIN_CLASS,
) -> KVCodecLMCacheStoragePluginConfig:
    """Build the dependency-free LMCache storage plugin launch config."""

    return KVCodecLMCacheStoragePluginConfig(
        enabled=enabled,
        plugin_name=plugin_name,
        module_path=module_path,
        class_name=class_name,
        artifact_root=artifact_root,
        codec=codec,
        quantization_axis=quantization_axis,
        layout_name=layout_name,
        layers_per_frame=layers_per_frame,
        head_rows=head_rows,
        head_cols=head_cols,
        dim_rows=dim_rows,
        dim_cols=dim_cols,
        io_threads=io_threads,
        nvenc_workers=nvenc_workers,
        nvdec_workers=nvdec_workers,
    )


def lmcache_kvcodec_storage_extra_config(
    plugin: KVCodecLMCacheStoragePluginConfig | None = None,
) -> dict[str, Any]:
    """Return ``extra_config`` keys consumed by LMCache ``storage_plugins``."""

    return (plugin or KVCodecLMCacheStoragePluginConfig()).extra_config()


def lmcache_kvcodec_storage_lmcache_config(
    plugin: KVCodecLMCacheStoragePluginConfig | None = None,
) -> dict[str, Any]:
    """Return LMCache EngineConfig fields for enabling the KVCodec plugin."""

    return (plugin or KVCodecLMCacheStoragePluginConfig()).lmcache_config()


def lmcache_kvcodec_storage_env(
    plugin: KVCodecLMCacheStoragePluginConfig | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return environment variables for LMCache/vLLM in-process plugin loading."""

    return (plugin or KVCodecLMCacheStoragePluginConfig()).env(base_env)

@dataclass(frozen=True, slots=True)
class VLLMKVTransferConfig:
    """Dependency-light representation of vLLM's ``KVTransferConfig`` JSON.

    vLLM consumes this object through ``--kv-transfer-config``.  Keeping it as
    plain JSON lets this package prepare real vLLM/LMCache launch arguments
    without importing either optional runtime at package import time.
    """

    kv_connector: str
    kv_role: KVRole
    kv_connector_extra_config: Mapping[str, Any] = field(default_factory=dict)
    kv_connector_module_path: str | None = None
    engine_id: str | None = None
    kv_rank: int | None = None
    kv_parallel_size: int | None = None
    kv_ip: str | None = None
    kv_port: int | None = None
    kv_buffer_device: str | None = None
    kv_buffer_size: float | None = None
    kv_load_failure_policy: Literal["recompute", "fail"] | None = None

    def __post_init__(self) -> None:
        if not self.kv_connector:
            raise ValueError("kv_connector must not be empty")
        if self.kv_role not in _VALID_KV_ROLES:
            raise ValueError(f"unsupported kv_role {self.kv_role!r}")
        if self.kv_connector_module_path is not None and not self.kv_connector_module_path:
            raise ValueError("kv_connector_module_path must not be empty when provided")
        if self.kv_rank is not None and self.kv_rank < 0:
            raise ValueError("kv_rank must be non-negative when provided")
        if self.kv_parallel_size is not None and self.kv_parallel_size <= 0:
            raise ValueError("kv_parallel_size must be positive when provided")
        if self.kv_port is not None and self.kv_port <= 0:
            raise ValueError("kv_port must be positive when provided")
        if self.kv_buffer_size is not None and self.kv_buffer_size <= 0:
            raise ValueError("kv_buffer_size must be positive when provided")
        if self.kv_load_failure_policy is not None and self.kv_load_failure_policy not in {"recompute", "fail"}:
            raise ValueError("kv_load_failure_policy must be 'recompute' or 'fail'")
        object.__setattr__(self, "kv_connector_extra_config", dict(self.kv_connector_extra_config))

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "kv_connector": self.kv_connector,
            "kv_role": self.kv_role,
        }
        if self.kv_connector_extra_config:
            config["kv_connector_extra_config"] = dict(self.kv_connector_extra_config)
        optional_values = {
            "kv_connector_module_path": self.kv_connector_module_path,
            "engine_id": self.engine_id,
            "kv_rank": self.kv_rank,
            "kv_parallel_size": self.kv_parallel_size,
            "kv_ip": self.kv_ip,
            "kv_port": self.kv_port,
            "kv_buffer_device": self.kv_buffer_device,
            "kv_buffer_size": self.kv_buffer_size,
            "kv_load_failure_policy": self.kv_load_failure_policy,
        }
        config.update({key: value for key, value in optional_values.items() if value is not None})
        return config

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

def instantiate_vllm_kv_transfer_config(config: VLLMKVTransferConfig | Mapping[str, Any]) -> Any:
    """Instantiate vLLM's real ``KVTransferConfig`` lazily.

    This is the dependency boundary between the lightweight config builder and
    an installed vLLM runtime. Importing ``cradle_codec.integration`` still
    does not import vLLM; callers opt in here when they want runtime validation.
    """

    try:
        from vllm.config import KVTransferConfig  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("vLLM is required to instantiate KVTransferConfig; install the serving optional dependencies") from exc
    data = config.to_dict() if isinstance(config, VLLMKVTransferConfig) else dict(config)
    return KVTransferConfig(**data)


def _normalized_server_urls(server_urls: Iterable[str]) -> list[str]:
    urls = [str(url).strip() for url in server_urls]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("server_urls must contain at least one non-empty URL")
    return urls


def _positive_float(name: str, value: float | None) -> float | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return value


def _mp_transfer_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = mode.strip().lower()
    if normalized not in {"auto", "engine_driven", "lmcache_driven"}:
        raise ValueError("mp_transfer_mode must be one of: auto, engine_driven, lmcache_driven")
    return normalized


def _lmcache_mp_extra_config(
    *,
    host: str,
    port: int,
    server_urls: Iterable[str] | None,
    mq_timeout_s: float | None,
    heartbeat_interval_s: float | None,
    mp_transfer_mode: str | None,
) -> dict[str, Any]:
    if port <= 0:
        raise ValueError("port must be positive")
    if server_urls is None:
        if not host:
            raise ValueError("host must not be empty")
        config: dict[str, Any] = {"lmcache.mp.host": host, "lmcache.mp.port": port}
    else:
        config = {"lmcache.mp.server_urls": _normalized_server_urls(server_urls)}
    if (timeout := _positive_float("mq_timeout_s", mq_timeout_s)) is not None:
        config["lmcache.mp.mq_timeout"] = timeout
    if (interval := _positive_float("heartbeat_interval_s", heartbeat_interval_s)) is not None:
        config["lmcache.mp.heartbeat_interval"] = interval
    if (mode := _mp_transfer_mode(mp_transfer_mode)) is not None:
        config["lmcache.mp.mp_transfer_mode"] = mode
    return config


def _merged_extra_config(base: Mapping[str, Any], extra_config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if extra_config:
        merged.update(extra_config)
    return merged


def lmcache_mp_kv_transfer_config(
    *,
    host: str = "tcp://localhost",
    port: int = 5555,
    server_urls: Iterable[str] | None = None,
    role: KVRole = "kv_both",
    use_lmcache_shipped_connector: bool = False,
    connector_module_path: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    engine_id: str | None = None,
    mq_timeout_s: float | None = None,
    heartbeat_interval_s: float | None = None,
    mp_transfer_mode: str | None = None,
) -> VLLMKVTransferConfig:
    """Build the real vLLM config for LMCache multi-process mode.

    ``host`` intentionally follows LMCache's documented connector spelling and
    should include a ZMQ transport prefix such as ``tcp://``. ``server_urls``
    enables the newer multi-server connector path and takes precedence over
    ``host``/``port`` when provided.
    """

    base_extra = _lmcache_mp_extra_config(
        host=host,
        port=port,
        server_urls=server_urls,
        mq_timeout_s=mq_timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
        mp_transfer_mode=mp_transfer_mode,
    )
    module_path = connector_module_path
    if module_path is None and use_lmcache_shipped_connector:
        module_path = _LMCACHE_MP_MODULE_PATH
    return VLLMKVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role=role,
        kv_connector_module_path=module_path,
        kv_connector_extra_config=_merged_extra_config(
            base_extra,
            extra_config,
        ),
        engine_id=engine_id,
    )


def lmcache_server_args(
    *,
    host: str = "localhost",
    port: int = 5555,
    http_port: int = 18080,
    l1_size_gb: float = 4.0,
    eviction_policy: Literal["LRU", "IsolatedLRU", "noop"] = "LRU",
    disable_logging: bool = True,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return argv for ``lmcache server`` without importing LMCache.

    LMCache 0.5.1 exposes the KVCodec ``storage_plugins`` hook through engine
    config/env rather than dedicated ``lmcache server`` flags; pair this argv
    with ``lmcache_kvcodec_storage_env()`` when launching the plugin.
    """

    if not host:
        raise ValueError("host must not be empty")
    if port <= 0:
        raise ValueError("port must be positive")
    if http_port <= 0:
        raise ValueError("http_port must be positive")
    if l1_size_gb <= 0:
        raise ValueError("l1_size_gb must be positive")
    if eviction_policy not in {"LRU", "IsolatedLRU", "noop"}:
        raise ValueError("eviction_policy must be one of: LRU, IsolatedLRU, noop")
    args = (
        "lmcache",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--http-port",
        str(http_port),
        "--l1-size-gb",
        str(l1_size_gb),
        "--eviction-policy",
        eviction_policy,
    )
    if disable_logging:
        args = (*args, "--disable-logging")
    return (*args, *tuple(extra_args))


def lmcache_vllm_launch_bundle(
    *,
    model: str = QWEN3_17B_MODEL,
    mode: Literal["mp", "in-process"] = "mp",
    vllm_port: int = 8000,
    lmcache_host: str = "tcp://localhost",
    lmcache_port: int = 5555,
    lmcache_http_port: int = 18080,
    lmcache_l1_size_gb: float = 4.0,
    role: KVRole = "kv_both",
    use_lmcache_shipped_connector: bool = False,
    storage_plugin: KVCodecLMCacheStoragePluginConfig | None = None,
    extra_vllm_args: Iterable[str] = (),
    extra_lmcache_args: Iterable[str] = (),
) -> dict[str, Any]:
    """Return env plus argv needed to launch vLLM with LMCache and KVCodec."""

    plugin = storage_plugin or KVCodecLMCacheStoragePluginConfig(enabled=False)
    if mode == "mp":
        kv_config = lmcache_mp_kv_transfer_config(
            host=lmcache_host,
            port=lmcache_port,
            role=role,
            use_lmcache_shipped_connector=use_lmcache_shipped_connector,
        )
        server_args: tuple[str, ...] | None = lmcache_server_args(
            host=lmcache_host.removeprefix("tcp://"),
            port=lmcache_port,
            http_port=lmcache_http_port,
            l1_size_gb=lmcache_l1_size_gb,
            extra_args=extra_lmcache_args,
        )
    elif mode == "in-process":
        kv_config = lmcache_inprocess_kv_transfer_config(role=role)
        server_args = None
    else:
        raise ValueError("mode must be 'mp' or 'in-process'")
    return {
        "env": lmcache_kvcodec_storage_env(plugin) if plugin.enabled else {},
        "lmcache_server_args": None if server_args is None else list(server_args),
        "vllm_serve_args": list(
            vllm_serve_args(
                model=model,
                port=vllm_port,
                kv_transfer_config=kv_config,
                extra_args=extra_vllm_args,
            )
        ),
    }


def lmcache_inprocess_kv_transfer_config(
    *,
    role: KVRole = "kv_both",
    use_native: bool | None = None,
    extra_config: Mapping[str, Any] | None = None,
    engine_id: str | None = None,
) -> VLLMKVTransferConfig:
    """Build the real vLLM config for in-process ``LMCacheConnectorV1`` mode."""

    base = {} if use_native is None else {"use_native": use_native}
    return VLLMKVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role=role,
        kv_connector_extra_config=_merged_extra_config(base, extra_config),
        engine_id=engine_id,
    )


def vllm_serve_args(
    *,
    model: str = QWEN3_17B_MODEL,
    port: int = 8000,
    kv_transfer_config: VLLMKVTransferConfig | Mapping[str, Any],
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return argv for ``vllm serve`` without importing vLLM."""

    if not model:
        raise ValueError("model must not be empty")
    if port <= 0:
        raise ValueError("port must be positive")
    if isinstance(kv_transfer_config, VLLMKVTransferConfig):
        config_json = kv_transfer_config.to_json()
    else:
        config_json = json.dumps(dict(kv_transfer_config), sort_keys=True, separators=(",", ":"))
    return (
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--kv-transfer-config",
        config_json,
        *tuple(extra_args),
    )


def qwen3_17b_lmcache_mp_serve_args(
    *,
    port: int = 8000,
    lmcache_host: str = "tcp://localhost",
    lmcache_port: int = 5555,
    lmcache_server_urls: Iterable[str] | None = None,
    use_lmcache_shipped_connector: bool = False,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return dependency-free vLLM argv for Qwen/Qwen3-1.7B + LMCache MP mode."""

    return vllm_serve_args(
        model=QWEN3_17B_MODEL,
        port=port,
        kv_transfer_config=lmcache_mp_kv_transfer_config(
            host=lmcache_host,
            port=lmcache_port,
            server_urls=lmcache_server_urls,
            use_lmcache_shipped_connector=use_lmcache_shipped_connector,
        ),
        extra_args=extra_args,
    )


def qwen3_17b_lmcache_inprocess_serve_args(
    *,
    port: int = 8000,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return dependency-free vLLM argv for Qwen/Qwen3-1.7B + in-process LMCache."""

    return vllm_serve_args(
        model=QWEN3_17B_MODEL,
        port=port,
        kv_transfer_config=lmcache_inprocess_kv_transfer_config(),
        extra_args=extra_args,
    )
