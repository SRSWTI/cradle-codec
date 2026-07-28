import unittest
import sys
import types

import numpy as np

from cradle_codec.integration import (
    CacheAxisOrder,
    IsolatedFetchScheduler,
    LMCacheArtifactKey,
    NumpyPagedKVCacheAdapter,
    PagedRestorePlan,
    RequestState,
    RuntimeRequest,
    SchedulerLimits,
    VLLMKVTransferConfig,
    instantiate_vllm_kv_transfer_config,
    is_lmcache_compatible_manifest,
    lmcache_artifact_dir_name,
    lmcache_mp_kv_transfer_config,
    normalize_lmcache_key,
    require_lmcache_compatible_manifest,
    restore_to_paged_cache,
)
from cradle_codec.manifest import (
    SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactPart,
    CodecManifest,
    KVShapeManifest,
    LayoutManifest,
    PartQuantizationManifest,
    QuantizationManifest,
)


def manifest_for(key: LMCacheArtifactKey, *, token_count: int | None = None) -> ArtifactManifest:
    quant = PartQuantizationManifest(
        mode="uint8_minmax",
        axis="channel",
        min_values=[0.0],
        scales=[1.0],
        source_dtype="float32",
        transport_dtype="uint8",
    )
    return ArtifactManifest(
        version=SCHEMA_VERSION,
        source_key=key.source_key,
        model=key.model,
        kv_shape=KVShapeManifest(2, 2, token_count or key.token_count or 4, 1, 2),
        layout=LayoutManifest(3, 1, 1, 1, 2),
        quantization=QuantizationManifest("uint8_minmax", "channel"),
        codec=CodecManifest("reference", "raw_reference", True, {}),
        parts=(
            ArtifactPart("k", 0, (0, 1), 0, token_count or key.token_count or 4, 1, 2, 1, 2, "parts/k.g0.bin", 0, "", quant),
            ArtifactPart("v", 0, (0, 1), 0, token_count or key.token_count or 4, 1, 2, 1, 2, "parts/v.g0.bin", 0, "", quant),
        ),
    )


class LMCacheCompatibilityTests(unittest.TestCase):
    def test_structured_lmcache_keys_are_stable(self) -> None:
        left = normalize_lmcache_key({"tokens": [3, 1], "tenant": "a"})
        right = normalize_lmcache_key({"tenant": "a", "tokens": [3, 1]})

        self.assertEqual(left, right)
        self.assertIn('"tokens":[3,1]', left)

    def test_lmcache_key_builds_manifest_compatible_source_key(self) -> None:
        key = LMCacheArtifactKey.from_lmcache_key(
            model="Qwen/Qwen3-8B",
            key=("tenant-a", 17),
            token_start=8,
            token_count=4,
        )
        manifest = manifest_for(key)

        self.assertTrue(is_lmcache_compatible_manifest(manifest, key))
        self.assertIs(require_lmcache_compatible_manifest(manifest, key), manifest)
        self.assertEqual(key.artifact_dir_name, lmcache_artifact_dir_name(model=key.model, key=("tenant-a", 17), token_start=8, token_count=4))
        self.assertTrue(key.artifact_dir_name.endswith(".kvcodec"))

    def test_lmcache_manifest_validation_rejects_wrong_span(self) -> None:
        key = LMCacheArtifactKey.from_lmcache_key(model="model", key="cache-key", token_count=4)
        manifest = manifest_for(key, token_count=3)

        self.assertFalse(is_lmcache_compatible_manifest(manifest, key))
        with self.assertRaisesRegex(ValueError, "token count"):
            require_lmcache_compatible_manifest(manifest, key)


class VLLMConfigTests(unittest.TestCase):
    def test_lmcache_mp_config_supports_multi_server_runtime_options(self) -> None:
        config = lmcache_mp_kv_transfer_config(
            server_urls=["http://127.0.0.1:9000", " http://127.0.0.1:9001 "],
            mq_timeout_s=2.5,
            heartbeat_interval_s=0.25,
            mp_transfer_mode="engine_driven",
        )

        data = config.to_dict()
        extra = data["kv_connector_extra_config"]
        self.assertEqual(data["kv_connector"], "LMCacheMPConnector")
        self.assertEqual(extra["lmcache.mp.server_urls"], ["http://127.0.0.1:9000", "http://127.0.0.1:9001"])
        self.assertEqual(extra["lmcache.mp.mq_timeout"], 2.5)
        self.assertEqual(extra["lmcache.mp.heartbeat_interval"], 0.25)
        self.assertEqual(extra["lmcache.mp.mp_transfer_mode"], "engine_driven")
        self.assertNotIn("lmcache.mp.host", extra)

    def test_lmcache_mp_config_rejects_empty_multi_server_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "server_urls"):
            lmcache_mp_kv_transfer_config(server_urls=["", "  "])

    def test_vllm_config_instantiation_is_lazy_and_exact(self) -> None:
        class FakeKVTransferConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        previous_vllm = sys.modules.get("vllm")
        previous_config = sys.modules.get("vllm.config")
        vllm_module = types.ModuleType("vllm")
        config_module = types.ModuleType("vllm.config")
        config_module.KVTransferConfig = FakeKVTransferConfig
        sys.modules["vllm"] = vllm_module
        sys.modules["vllm.config"] = config_module
        try:
            config = VLLMKVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both", kv_connector_extra_config={"use_native": False})
            instantiated = instantiate_vllm_kv_transfer_config(config)
        finally:
            if previous_vllm is None:
                sys.modules.pop("vllm", None)
            else:
                sys.modules["vllm"] = previous_vllm
            if previous_config is None:
                sys.modules.pop("vllm.config", None)
            else:
                sys.modules["vllm.config"] = previous_config

        self.assertIsInstance(instantiated, FakeKVTransferConfig)
        self.assertEqual(instantiated.kwargs, config.to_dict())

class PagedRestoreAdapterTests(unittest.TestCase):
    def test_restore_maps_token_positions_to_physical_slots(self) -> None:
        restored = np.arange(2 * 3 * 4 * 2 * 2, dtype=np.float32).reshape(2, 3, 4, 2, 2)
        cache = np.full((2, 3, 8, 2, 2), -1.0, dtype=np.float32)
        plan = PagedRestorePlan.from_slots([5, 2], token_start=1)

        result = restore_to_paged_cache(cache, restored, plan)

        self.assertEqual(result.tokens_restored, 2)
        self.assertEqual(result.slots_written, (5, 2))
        self.assertTrue(np.array_equal(cache[:, :, 5, :, :], restored[:, :, 1, :, :]))
        self.assertTrue(np.array_equal(cache[:, :, 2, :, :], restored[:, :, 2, :, :]))
        self.assertTrue(np.all(cache[:, :, 0, :, :] == -1.0))

    def test_restore_honors_caller_axis_order_and_layer_subset(self) -> None:
        restored = np.arange(2 * 3 * 3 * 1 * 2, dtype=np.float32).reshape(2, 3, 3, 1, 2)
        cache = np.full((6, 2, 3, 1, 2), -1.0, dtype=np.float32)
        axes = CacheAxisOrder(slot_axis=0, side_axis=1, layer_axis=2, kv_head_axis=3, head_dim_axis=4)
        plan = PagedRestorePlan.from_slots([4], token_start=2, layer_indices=[1])
        adapter = NumpyPagedKVCacheAdapter(cache, axes=axes)

        result = adapter.restore(restored, plan)

        self.assertEqual(result.layers_written, (1,))
        self.assertTrue(np.array_equal(cache[4, :, 1, :, :], restored[:, 1, 2, :, :]))
        self.assertTrue(np.all(cache[4, :, 0, :, :] == -1.0))
        self.assertTrue(np.all(cache[4, :, 2, :, :] == -1.0))

    def test_restore_rejects_duplicate_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PagedRestorePlan.from_slots([1, 1])


class IsolatedFetchSchedulerTests(unittest.TestCase):
    def test_no_fetch_request_starts_while_older_fetch_request_waits_on_fetch(self) -> None:
        scheduler = IsolatedFetchScheduler(SchedulerLimits(max_concurrent_fetches=1, max_running_no_fetch=1, max_running_fetch=1))
        scheduler.submit(RuntimeRequest("fetch-first", fetch_key="artifact/a"))
        scheduler.submit(RuntimeRequest("plain-second"))

        step = scheduler.advance()

        self.assertEqual(step.started_fetches, ("fetch-first",))
        self.assertEqual(step.started_running, ("plain-second",))
        self.assertEqual(scheduler.state("fetch-first"), RequestState.FETCHING)
        self.assertEqual(scheduler.state("plain-second"), RequestState.RUNNING)

    def test_fetch_request_runs_only_after_fetch_completion(self) -> None:
        scheduler = IsolatedFetchScheduler(SchedulerLimits(max_concurrent_fetches=1, max_running_no_fetch=1, max_running_fetch=1))
        scheduler.submit(RuntimeRequest("fetch", fetch_key="artifact/a"))
        scheduler.advance()

        step = scheduler.advance(fetch_completed=["fetch"])

        self.assertEqual(step.started_running, ("fetch",))
        self.assertEqual(scheduler.state("fetch"), RequestState.RUNNING)

    def test_fetch_lane_is_fifo_and_capacity_limited(self) -> None:
        scheduler = IsolatedFetchScheduler(SchedulerLimits(max_concurrent_fetches=1, max_running_no_fetch=0, max_running_fetch=0))
        scheduler.submit(RuntimeRequest("fetch-a", fetch_key="artifact/a"))
        scheduler.submit(RuntimeRequest("fetch-b", fetch_key="artifact/b"))

        first = scheduler.advance()
        second = scheduler.advance(fetch_completed=["fetch-a"])

        self.assertEqual(first.started_fetches, ("fetch-a",))
        self.assertEqual(second.started_fetches, ("fetch-b",))
        self.assertEqual(scheduler.state("fetch-a"), RequestState.READY)
        self.assertEqual(scheduler.state("fetch-b"), RequestState.FETCHING)

    def test_completed_running_requests_free_their_own_lane(self) -> None:
        scheduler = IsolatedFetchScheduler(SchedulerLimits(max_concurrent_fetches=0, max_running_no_fetch=1, max_running_fetch=1))
        scheduler.submit(RuntimeRequest("plain-a"))
        scheduler.submit(RuntimeRequest("plain-b"))
        scheduler.advance()

        step = scheduler.advance(completed=["plain-a"])

        self.assertEqual(scheduler.state("plain-a"), RequestState.FINISHED)
        self.assertEqual(step.started_running, ("plain-b",))
        self.assertEqual(scheduler.state("plain-b"), RequestState.RUNNING)


if __name__ == "__main__":
    unittest.main()
