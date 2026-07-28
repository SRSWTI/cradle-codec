import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.metadata import LMCacheMetadata
except ImportError:  # pragma: no cover - serving extra is optional.
    torch = None
    CacheEngineKey = None
    MemoryFormat = None
    LMCacheMetadata = None

from cradle_codec.integration.lmcache_storage import (
    KVCodecLMCacheStorageBackend,
    _kv_array_to_lmcache_array,
    _lmcache_tensor_to_kv_array,
)


class _FakeConfig:
    cache_policy = "LRU"
    local_disk = None
    max_local_disk_size = 0.0
    storage_plugins = ["kvcodec"]

    def __init__(self, artifact_root: Path, *, layout_name: str | None = None):
        self.extra_config = {
            "storage_plugin.kvcodec.module_path": "cradle_codec.integration.lmcache_storage",
            "storage_plugin.kvcodec.class_name": "KVCodecLMCacheStorageBackend",
            "storage_plugin.kvcodec.artifact_root": str(artifact_root),
            "storage_plugin.kvcodec.codec": "reference",
            "storage_plugin.kvcodec.layers_per_frame": 1,
        }
        if layout_name is not None:
            self.extra_config["storage_plugin.kvcodec.layout_name"] = layout_name


class _FakeMemoryObj:
    def __init__(self, tensor, *, fmt, cached_positions=None):
        self._tensor = tensor
        self.metadata = types.SimpleNamespace(
            shape=torch.Size(tensor.shape),
            dtype=tensor.dtype,
            fmt=fmt,
            cached_positions=cached_positions,
        )
        self.ref_count_down_called = False

    @property
    def tensor(self):
        return self._tensor

    def get_physical_size(self):
        return self._tensor.numel() * self._tensor.element_size()

    def ref_count_down(self):
        self.ref_count_down_called = True


class _FakeLocalCPUBackend:
    def __init__(self):
        self.allocations = []

    def allocate(self, shape, dtype, fmt, *args, **kwargs):
        del args, kwargs
        obj = _FakeMemoryObj(torch.empty(tuple(shape), dtype=dtype), fmt=fmt)
        self.allocations.append(obj)
        return obj


class LMCacheLayoutConfigTests(unittest.TestCase):
    def test_layout_builder_honors_named_layout(self):
        backend = object.__new__(KVCodecLMCacheStorageBackend)
        backend.config = _FakeConfig(Path("unused"), layout_name="h2x1_d1x3_2x3")
        backend._plugin_names = ("kvcodec",)
        backend.layers_per_frame = 1
        backend.layout_name = "h2x1_d1x3_2x3"
        kv = np.empty((2, 2, 4, 2, 3), dtype=np.float16)

        layout = backend._layout_for(kv)

        self.assertEqual((layout.tiling.head_rows, layout.tiling.head_cols), (2, 1))
        self.assertEqual((layout.tiling.dim_rows, layout.tiling.dim_cols), (1, 3))


@unittest.skipIf(torch is None, "LMCache serving extra is not installed")
class LMCacheKVCodecStorageBackendTests(unittest.TestCase):
    def metadata(self):
        return LMCacheMetadata(
            model_name="synthetic",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(2, 2, 4, 2, 3),
            chunk_size=4,
        )

    def test_kv_2ltd_conversion_uses_lmcache_kv_shape_heads_and_head_size(self):
        metadata = self.metadata()
        lmcache_tensor = torch.arange(2 * 2 * 4 * 6, dtype=torch.float16).reshape(2, 2, 4, 6)

        kv = _lmcache_tensor_to_kv_array(lmcache_tensor, metadata)
        restored = _kv_array_to_lmcache_array(kv, lmcache_tensor.shape)

        self.assertEqual(kv.shape, (2, 2, 4, 2, 3))
        self.assertTrue(torch.equal(torch.from_numpy(restored), lmcache_tensor))

        bfloat_tensor = lmcache_tensor.to(torch.bfloat16)
        bfloat_kv = _lmcache_tensor_to_kv_array(bfloat_tensor, metadata)
        bfloat_restored = torch.from_numpy(_kv_array_to_lmcache_array(bfloat_kv, bfloat_tensor.shape)).to(torch.bfloat16)
        self.assertEqual(bfloat_kv.dtype.name, "float32")
        self.assertTrue(torch.equal(bfloat_restored, bfloat_tensor))

    def test_backend_honors_named_layout(self):
        metadata = self.metadata()
        with tempfile.TemporaryDirectory() as tmp:
            backend = KVCodecLMCacheStorageBackend(
                config=_FakeConfig(Path(tmp), layout_name="h2x1_d1x3_2x3"),
                dst_device="cpu",
                metadata=metadata,
                local_cpu_backend=_FakeLocalCPUBackend(),
                loop=None,
            )
            kv = np.empty((2, 2, 4, 2, 3), dtype=np.float16)

            layout = backend._layout_for(kv)

            self.assertEqual((layout.tiling.head_rows, layout.tiling.head_cols), (2, 1))
            self.assertEqual((layout.tiling.dim_rows, layout.tiling.dim_cols), (1, 3))
            backend.close()

    def test_backend_stores_artifact_index_and_loads_memory_obj(self):
        metadata = self.metadata()
        cached_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        tensor = (torch.arange(2 * 2 * 4 * 6, dtype=torch.float32).reshape(2, 2, 4, 6) / 17).to(torch.float16)
        source_obj = _FakeMemoryObj(tensor, fmt=MemoryFormat.KV_2LTD, cached_positions=cached_positions)
        key = CacheEngineKey("synthetic", 1, 0, 1234, torch.float16)

        with tempfile.TemporaryDirectory() as tmp:
            backend = KVCodecLMCacheStorageBackend(
                config=_FakeConfig(Path(tmp)),
                dst_device="cpu",
                metadata=metadata,
                local_cpu_backend=_FakeLocalCPUBackend(),
                loop=None,
            )
            backend.batched_submit_put_task([key], [source_obj])

            self.assertTrue(backend.contains(key))
            self.assertTrue((Path(tmp) / "index.json").exists())
            self.assertEqual(len([path for path in Path(tmp).iterdir() if path.suffix == ".kvcodec"]), 1)

            restored_obj = backend.get_blocking(key)
            self.assertIsNotNone(restored_obj)
            self.assertEqual(tuple(restored_obj.tensor.shape), tuple(tensor.shape))
            self.assertEqual(restored_obj.tensor.dtype, torch.float16)
            self.assertTrue(torch.equal(restored_obj.metadata.cached_positions, cached_positions))
            self.assertLess(float((restored_obj.tensor - tensor).abs().max()), 0.01)
            backend.close()


if __name__ == "__main__":
    unittest.main()
