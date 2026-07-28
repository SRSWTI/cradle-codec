from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from cradle_codec.codec import FFmpegHEVCCodec, FrameCodec, PyNvVideoCodecHEVCCodec, RawReferenceCodec
from cradle_codec.integration.lmcache import lmcache_source_key
from cradle_codec.layout import HeadDimTiling, KVCodecLayout, layout_from_name, validate_layout
from cradle_codec.pipeline import decode_kv_artifact, encode_kv_chunk
from cradle_codec.quant import QuantizationSpec
from cradle_codec.store import artifact_dir_name

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when LMCache serving extra is installed.
    import torch
    from lmcache import torch_dev, torch_device_type
    from lmcache.utils import CacheEngineKey, DiskCacheMetadata
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat, MemoryObj
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
    from lmcache.v1.storage_backend.cache_policy import get_cache_policy
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
except ImportError as exc:  # pragma: no cover - import-time guard for non-serving installs.
    _LMCACHE_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    torch_dev = None  # type: ignore[assignment]
    torch_device_type = "cpu"  # type: ignore[assignment]
    CacheEngineKey = Any  # type: ignore[misc,assignment]
    DiskCacheMetadata = Any  # type: ignore[misc,assignment]
    LMCacheEngineConfig = Any  # type: ignore[misc,assignment]
    MemoryFormat = Any  # type: ignore[misc,assignment]
    MemoryObj = Any  # type: ignore[misc,assignment]
    LMCacheMetadata = Any  # type: ignore[misc,assignment]
    LocalCPUBackend = Any  # type: ignore[misc,assignment]

    class StorageBackendInterface:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("LMCache is required for KVCodecLMCacheStorageBackend") from _LMCACHE_IMPORT_ERROR

else:
    _LMCACHE_IMPORT_ERROR = None


_INDEX_VERSION = 1
_DEFAULT_ARTIFACT_ROOT = ".cradle-codec"
_DEFAULT_CODEC = "reference"
_DEFAULT_QUANTIZATION_AXIS = "channel"
_DEFAULT_LAYERS_PER_FRAME = 3
_DEFAULT_IO_THREADS = 1
_PLUGIN_CLASS_NAMES = {"KVCodecLMCacheStorageBackend", "LMCacheKVCodecStorageBackend"}


@dataclass(frozen=True)
class KVCodecLMCacheStoredEntry:
    """Persisted metadata for one LMCache CacheEngineKey-backed artifact."""

    key: str
    artifact_dir: str
    source_key: str
    model: str
    size: int
    shape: tuple[int, ...]
    dtype: str
    fmt: str
    cached_positions: dict[str, Any] | None


class KVCodecLMCacheStorageBackend(StorageBackendInterface):
    """LMCache storage plugin that persists KV_2LTD chunks as KVCodec artifacts.

    LMCache loads this class through ``storage_plugins`` with constructor
    arguments ``config, dst_device, metadata, local_cpu_backend, loop``.  Each
    stored ``CacheEngineKey`` gets a dedicated artifact directory and an index
    entry carrying enough LMCache metadata to allocate and refill a
    ``MemoryObj`` on reads.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        dst_device: str = torch_device_type,
        metadata: LMCacheMetadata | None = None,
        local_cpu_backend: LocalCPUBackend | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if _LMCACHE_IMPORT_ERROR is not None:
            raise ImportError("LMCache is required for KVCodecLMCacheStorageBackend") from _LMCACHE_IMPORT_ERROR
        if metadata is None:
            raise ValueError("KVCodecLMCacheStorageBackend requires LMCache metadata")
        if local_cpu_backend is None:
            raise ValueError("KVCodecLMCacheStorageBackend requires LocalCPUBackend for retrieval allocation")

        if torch_dev is not None and torch_dev.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        self.dst_device = dst_device
        self._plugin_names = tuple(_matching_plugin_names(config))

        self.artifact_root = Path(
            str(_config_value(config, self._plugin_names, "artifact_root", default=_default_artifact_root(config)))
        ).expanduser()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifact_root / "index.json"

        self.codec_name = str(_config_value(config, self._plugin_names, "codec", default=_DEFAULT_CODEC))
        self.quantization_axis = str(
            _config_value(config, self._plugin_names, "quantization_axis", default=_DEFAULT_QUANTIZATION_AXIS)
        )
        configured_layout = _config_value(config, self._plugin_names, "layout_name", default=None)
        self.layout_name = None if configured_layout is None else str(configured_layout).strip() or None
        self.layers_per_frame = int(
            _config_value(config, self._plugin_names, "layers_per_frame", default=_DEFAULT_LAYERS_PER_FRAME)
        )
        self.io_threads = max(1, int(_config_value(config, self._plugin_names, "io_threads", default=_DEFAULT_IO_THREADS)))
        self.nvenc_workers = max(1, int(_config_value(config, self._plugin_names, "nvenc_workers", default=1)))
        self.nvdec_workers = max(1, int(_config_value(config, self._plugin_names, "nvdec_workers", default=1)))

        self.cache_policy = get_cache_policy(getattr(config, "cache_policy", "LRU"))
        self.dict = self.cache_policy.init_mutable_mapping()
        self.keys_in_request: list[CacheEngineKey] = []
        self._lock = threading.RLock()
        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()
        self._executor = ThreadPoolExecutor(max_workers=self.io_threads, thread_name_prefix="lmcache-kvcodec")
        self._closed = False
        self.current_cache_size = 0
        self.max_cache_size = _max_cache_size_bytes(config, self._plugin_names)

        self._load_index()

    def __str__(self) -> str:
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self._lock:
            meta = self.dict.get(key)
            if meta is None:
                return False
            if not (Path(meta.path) / "manifest.json").exists():
                self._remove_locked(key, force=True, missing_ok=True)
                return False
            if pin:
                meta.pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self) -> None:
        with self._lock:
            for key in reversed(self.keys_in_request):
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    def pin(self, key: CacheEngineKey) -> bool:
        with self._lock:
            meta = self.dict.get(key)
            if meta is None:
                return False
            meta.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self._lock:
            meta = self.dict.get(key)
            if meta is None:
                return False
            meta.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self._lock:
            return self._remove_locked(key, force=force, missing_ok=False)

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        self.batched_submit_put_task([key], [memory_obj], on_complete_callback=on_complete_callback)
        return None

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> list[Future] | None:
        del transfer_spec
        for key, obj in zip(keys, objs, strict=False):
            self._store_one(key, obj)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as exc:  # noqa: BLE001 - callback boundary mirrors LMCache backends.
                    logger.warning("on_complete_callback failed for key %s: %s", key, exc)
        return None

    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        running_loop = asyncio.get_running_loop()
        await running_loop.run_in_executor(
            self._executor,
            self.batched_submit_put_task,
            keys,
            objs,
            transfer_spec,
            on_complete_callback,
        )

    def get_blocking(self, key: CacheEngineKey) -> MemoryObj | None:
        with self._lock:
            meta = self.dict.get(key)
            if meta is None:
                return None
            artifact_dir = Path(meta.path)
            shape = meta.shape
            dtype = meta.dtype
            fmt = meta.fmt
            cached_positions = _clone_tensor(meta.cached_positions)

        if fmt is not MemoryFormat.KV_2LTD:
            raise ValueError(f"KVCodec LMCache backend only loads KV_2LTD entries, got {fmt}")
        if dtype is None or shape is None:
            raise ValueError(f"KVCodec LMCache index entry for {key.to_string()} is missing dtype or shape")

        kv = decode_kv_artifact(artifact_dir, codec=self._codec_for_decode())
        lmcache_array = _kv_array_to_lmcache_array(kv, shape)
        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            return None
        tensor = memory_obj.tensor
        if tensor is None:
            memory_obj.ref_count_down()
            return None
        restored = torch.as_tensor(lmcache_array, dtype=dtype, device=tensor.device)
        tensor.copy_(restored)
        memory_obj.metadata.cached_positions = cached_positions

        with self._lock:
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
        return memory_obj

    def get_non_blocking(self, key: CacheEngineKey, location: Optional[str] = None) -> Future:
        del location
        return self._executor.submit(self.get_blocking, key)

    async def batched_async_contains(self, lookup_id: str, keys: list[CacheEngineKey], pin: bool = False) -> int:
        del lookup_id
        with self._lock:
            hit_chunks = 0
            for key in keys:
                meta = self.dict.get(key)
                if meta is None:
                    return hit_chunks
                if not (Path(meta.path) / "manifest.json").exists():
                    self._remove_locked(key, force=True, missing_ok=True)
                    return hit_chunks
                if pin:
                    meta.pin()
                    self.keys_in_request.append(key)
                hit_chunks += 1
            return hit_chunks

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        del lookup_id, transfer_spec
        running_loop = asyncio.get_running_loop()
        futures = [running_loop.run_in_executor(self._executor, self.get_blocking, key) for key in keys]
        results = await asyncio.gather(*futures)
        return [obj for obj in results if obj is not None]

    def get_allocator_backend(self) -> LocalCPUBackend:
        return self.local_cpu_backend

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._write_index_locked()
        self._executor.shutdown(wait=True)

    def _store_one(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        if memory_obj.tensor is None:
            raise ValueError("cannot store LMCache MemoryObj without a tensor")
        fmt = memory_obj.metadata.fmt
        if fmt is not MemoryFormat.KV_2LTD:
            raise ValueError(f"KVCodec LMCache backend only stores KV_2LTD, got {fmt}")

        with self._put_lock:
            if key in self._put_tasks:
                return
            self._put_tasks.add(key)

        try:
            lmcache_shape = tuple(int(dim) for dim in memory_obj.metadata.shape)
            dtype = memory_obj.metadata.dtype
            if dtype is None:
                raise ValueError("cannot store LMCache MemoryObj without dtype metadata")
            kv = _lmcache_tensor_to_kv_array(memory_obj.tensor, self.metadata)
            layout = self._layout_for(kv)
            source_key = lmcache_source_key(
                model=self._model_name(key),
                key=key.to_string(),
                token_start=0,
                token_count=kv.shape[2],
                namespace="lmcache-kvcodec",
            )
            final_dir = self.artifact_root / artifact_dir_name(source_key)

            self._reserve_space(key, memory_obj.get_physical_size())
            with tempfile.TemporaryDirectory(prefix=".tmp-", dir=self.artifact_root) as tmp:
                tmp_dir = Path(tmp)
                encode_kv_chunk(
                    kv,
                    tmp_dir,
                    source_key=source_key,
                    model=self._model_name(key),
                    layout=layout,
                    quantization=QuantizationSpec(mode="uint8_minmax", axis=self.quantization_axis),
                    codec=self._codec_for_encode(),
                )
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                tmp_dir.rename(final_dir)

            size = _directory_size(final_dir)
            cached_positions = _serialize_tensor(memory_obj.metadata.cached_positions)
            with self._lock:
                previous = self.dict.get(key)
                if previous is not None:
                    self.current_cache_size -= int(previous.size)
                self.current_cache_size += size
                self.dict[key] = DiskCacheMetadata(
                    str(final_dir),
                    size,
                    torch.Size(lmcache_shape),
                    dtype,
                    _deserialize_tensor(cached_positions),
                    fmt,
                    0,
                )
                self.cache_policy.update_on_put(key)
                self._write_index_locked()
        finally:
            with self._put_lock:
                self._put_tasks.discard(key)

    def _layout_for(self, kv: np.ndarray) -> KVCodecLayout:
        num_layers = int(kv.shape[1])
        num_kv_heads = int(kv.shape[3])
        head_dim = int(kv.shape[4])
        layers_per_frame = min(self.layers_per_frame, 3, num_layers)
        if self.layout_name is not None:
            return layout_from_name(
                self.layout_name,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layers_per_frame=layers_per_frame,
            )
        head_rows = int(_config_value(self.config, self._plugin_names, "head_rows", default=1))
        head_cols = int(_config_value(self.config, self._plugin_names, "head_cols", default=num_kv_heads))
        dim_rows = int(_config_value(self.config, self._plugin_names, "dim_rows", default=1))
        dim_cols = int(_config_value(self.config, self._plugin_names, "dim_cols", default=head_dim))
        layout = KVCodecLayout(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layers_per_frame=layers_per_frame,
            tiling=HeadDimTiling(
                head_rows=head_rows,
                head_cols=head_cols,
                dim_rows=dim_rows,
                dim_cols=dim_cols,
            ),
        )
        return validate_layout(layout)

    def _codec_for_encode(self) -> FrameCodec:
        normalized = self.codec_name.strip().lower().replace("_", "-")
        if normalized in {"reference", "raw", "raw-reference"}:
            return RawReferenceCodec()
        if normalized in {"ffmpeg", "ffmpeg-hevc", "libx265"}:
            return FFmpegHEVCCodec()
        if normalized in {"pynv", "pynvvideocodec", "pynv-hevc", "nvenc"}:
            return self._pynv_codec()
        raise ValueError("KVCodec LMCache storage codec must be one of: reference, ffmpeg, pynvvideocodec")

    def _codec_for_decode(self) -> FrameCodec | None:
        normalized = self.codec_name.strip().lower().replace("_", "-")
        if normalized in {"reference", "raw", "raw-reference"}:
            return RawReferenceCodec()
        if normalized in {"ffmpeg", "ffmpeg-hevc", "libx265"}:
            return FFmpegHEVCCodec()
        if normalized in {"pynv", "pynvvideocodec", "pynv-hevc", "nvenc"}:
            return self._pynv_codec()
        return None

    def _pynv_codec(self) -> PyNvVideoCodecHEVCCodec:
        return PyNvVideoCodecHEVCCodec(nvenc_workers=self.nvenc_workers, nvdec_workers=self.nvdec_workers)

    def _model_name(self, key: CacheEngineKey) -> str:
        return str(getattr(self.metadata, "served_model_name", None) or getattr(self.metadata, "model_name", None) or key.model_name)

    def _reserve_space(self, new_key: CacheEngineKey, incoming_size: int) -> None:
        if self.max_cache_size <= 0:
            return
        with self._lock:
            while self.current_cache_size + incoming_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(self.dict, num_candidates=1)
                evict_keys = [key for key in evict_keys if key != new_key]
                if not evict_keys:
                    raise RuntimeError("KVCodec LMCache artifact root is full and no evictable entries are available")
                for evict_key in evict_keys:
                    self._remove_locked(evict_key, force=False, missing_ok=True)

    def _remove_locked(self, key: CacheEngineKey, *, force: bool, missing_ok: bool) -> bool:
        meta = self.dict.get(key)
        if meta is None:
            return False
        if not force and getattr(meta, "is_pinned", False):
            return False
        self.dict.pop(key, None)
        self.current_cache_size -= int(meta.size)
        shutil.rmtree(meta.path, ignore_errors=missing_ok)
        if force:
            self.cache_policy.update_on_force_evict(key)
        self._write_index_locked()
        return True

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        data = json.loads(self.index_path.read_text())
        if int(data.get("version", 0)) != _INDEX_VERSION:
            return
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            return
        with self._lock:
            for key_str, payload in entries.items():
                try:
                    key = CacheEngineKey.from_string(key_str)
                    entry = _entry_from_json(key_str, payload)
                    artifact_path = self.artifact_root / entry.artifact_dir
                    if not (artifact_path / "manifest.json").exists():
                        continue
                    dtype = _torch_dtype_from_string(entry.dtype)
                    fmt = _memory_format_from_string(entry.fmt)
                    cached_positions = _deserialize_tensor(entry.cached_positions)
                    self.dict[key] = DiskCacheMetadata(
                        str(artifact_path),
                        entry.size,
                        torch.Size(entry.shape),
                        dtype,
                        cached_positions,
                        fmt,
                        0,
                    )
                    self.current_cache_size += entry.size
                except Exception:
                    continue

    def _write_index_locked(self) -> None:
        entries: dict[str, dict[str, Any]] = {}
        for key, meta in self.dict.items():
            key_str = key.to_string()
            artifact_path = Path(meta.path)
            entries[key_str] = {
                "artifact_dir": artifact_path.name,
                "source_key": lmcache_source_key(
                    model=self._model_name(key),
                    key=key_str,
                    token_start=0,
                    token_count=int(meta.shape[MemoryFormat.KV_2LTD.token_dim()]) if meta.shape is not None else None,
                    namespace="lmcache-kvcodec",
                ),
                "model": self._model_name(key),
                "size": int(meta.size),
                "shape": list(meta.shape) if meta.shape is not None else None,
                "dtype": _torch_dtype_to_string(meta.dtype),
                "fmt": _memory_format_to_string(meta.fmt),
                "cached_positions": _serialize_tensor(meta.cached_positions),
            }
        data = {"version": _INDEX_VERSION, "entries": entries}
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
        tmp.replace(self.index_path)


LMCacheKVCodecStorageBackend = KVCodecLMCacheStorageBackend


def _matching_plugin_names(config: LMCacheEngineConfig) -> Iterable[str]:
    extra = getattr(config, "extra_config", None) or {}
    for name in getattr(config, "storage_plugins", None) or ():
        module_path = extra.get(f"storage_plugin.{name}.module_path")
        class_name = extra.get(f"storage_plugin.{name}.class_name")
        if module_path == __name__ or class_name in _PLUGIN_CLASS_NAMES:
            yield str(name)


def _config_value(config: LMCacheEngineConfig, plugin_names: Sequence[str], key: str, *, default: Any) -> Any:
    extra = getattr(config, "extra_config", None) or {}
    for plugin_name in plugin_names:
        scoped_key = f"storage_plugin.{plugin_name}.{key}"
        if scoped_key in extra:
            return extra[scoped_key]
    legacy_keys = (f"kvcodec_lmcache_{key}", f"kvcodec_{key}")
    for legacy_key in legacy_keys:
        if legacy_key in extra:
            return extra[legacy_key]
    return default


def _default_artifact_root(config: LMCacheEngineConfig) -> str:
    local_disk = getattr(config, "local_disk", None)
    if isinstance(local_disk, str) and local_disk.strip():
        return local_disk.split(",", 1)[0].strip()
    return _DEFAULT_ARTIFACT_ROOT


def _max_cache_size_bytes(config: LMCacheEngineConfig, plugin_names: Sequence[str]) -> int:
    configured = _config_value(config, plugin_names, "max_cache_bytes", default=None)
    if configured is not None:
        return int(configured)
    max_gib = float(getattr(config, "max_local_disk_size", 0.0) or 0.0)
    return int(max_gib * 1024**3)


def _lmcache_tensor_to_kv_array(tensor: Any, metadata: LMCacheMetadata) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"expected LMCache KV_2LTD rank 4 [2,L,T,hidden_dim], got rank {tensor.ndim}")
    num_layers, num_sides, _chunk_tokens, num_heads, head_size = metadata.kv_shape
    if int(tensor.shape[0]) != int(num_sides) or int(tensor.shape[1]) != int(num_layers):
        raise ValueError(
            "LMCache tensor shape does not match metadata.kv_shape: "
            f"tensor sides/layers={tuple(tensor.shape[:2])}, metadata sides/layers={(num_sides, num_layers)}"
        )
    hidden_dim = int(num_heads) * int(head_size)
    if int(tensor.shape[3]) != hidden_dim:
        raise ValueError(f"expected hidden_dim {hidden_dim} from LMCache metadata.kv_shape, got {int(tensor.shape[3])}")
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    if cpu_tensor.dtype == torch.bfloat16:
        array = cpu_tensor.to(torch.float32).numpy()
    else:
        array = cpu_tensor.numpy()
    return np.ascontiguousarray(array.reshape(num_sides, num_layers, int(tensor.shape[2]), num_heads, head_size))


def _kv_array_to_lmcache_array(kv: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if kv.ndim != 5:
        raise ValueError(f"expected canonical KV rank 5 [2,L,T,H,D], got rank {kv.ndim}")
    expected = tuple(int(dim) for dim in shape)
    if len(expected) != 4:
        raise ValueError(f"expected LMCache KV_2LTD rank 4 shape metadata, got {expected}")
    flat = np.ascontiguousarray(kv.reshape(kv.shape[0], kv.shape[1], kv.shape[2], kv.shape[3] * kv.shape[4]))
    if tuple(flat.shape) != expected:
        raise ValueError(f"decoded KV shape {tuple(flat.shape)} does not match LMCache index shape {expected}")
    return flat


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _torch_dtype_to_string(dtype: Any) -> str:
    if dtype is None:
        return ""
    return str(dtype)


def _torch_dtype_from_string(value: str) -> Any:
    if not value:
        return None
    name = value.replace("torch.", "")
    if name == "half":
        name = "float16"
    if name == "float":
        name = "float32"
    if name == "double":
        name = "float64"
    return getattr(torch, name)


def _memory_format_to_string(fmt: Any) -> str:
    if fmt is None:
        return ""
    return getattr(fmt, "name", str(fmt))


def _memory_format_from_string(value: str) -> Any:
    if not value:
        return MemoryFormat.UNDEFINED
    name = value.split(".")[-1]
    return MemoryFormat[name]


def _serialize_tensor(tensor: Any) -> dict[str, Any] | None:
    if tensor is None:
        return None
    cpu_tensor = tensor.detach().to("cpu").contiguous()
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "values": cpu_tensor.tolist(),
    }


def _deserialize_tensor(data: dict[str, Any] | None) -> Any:
    if data is None:
        return None
    dtype = _torch_dtype_from_string(str(data["dtype"]))
    tensor = torch.tensor(data["values"], dtype=dtype)
    return tensor.reshape(tuple(int(dim) for dim in data.get("shape", tensor.shape)))


def _clone_tensor(tensor: Any) -> Any:
    return None if tensor is None else tensor.detach().clone()


def _entry_from_json(key: str, payload: dict[str, Any]) -> KVCodecLMCacheStoredEntry:
    shape = payload.get("shape")
    if shape is None:
        raise ValueError("index entry is missing shape")
    return KVCodecLMCacheStoredEntry(
        key=key,
        artifact_dir=str(payload["artifact_dir"]),
        source_key=str(payload.get("source_key", "")),
        model=str(payload.get("model", "")),
        size=int(payload["size"]),
        shape=tuple(int(dim) for dim in shape),
        dtype=str(payload.get("dtype", "")),
        fmt=str(payload.get("fmt", "")),
        cached_positions=payload.get("cached_positions"),
    )
