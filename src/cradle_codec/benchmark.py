from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import queue
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np

from cradle_codec.fetch import select_variant
from cradle_codec.layout import unpack_frame_batches_to_kv
from cradle_codec.manifest import read_manifest, verify_part_payload
from cradle_codec.pipeline import decode_part_payload, select_manifest_parts


@dataclass(frozen=True)
class BenchmarkMetric:
    method: str
    ttft_ms: float
    network_bytes: int
    decode_ms: float
    restore_ms: float
    scheduler_wait_ms: float
    encoded_to_raw_ratio: float | None = None
    raw_to_encoded_ratio: float | None = None
    max_abs_error: float | None = None
    selected_variant: str | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    source_key: str
    model: str
    raw_bytes: int
    bandwidth_bytes_per_sec: float
    metrics: tuple[BenchmarkMetric, ...]

    def to_json(self) -> str:
        return json.dumps(benchmark_report_to_dict(self), sort_keys=True, indent=2)


def _positive_float(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _network_ms(bytes_count: int, bandwidth_bytes_per_sec: float) -> float:
    return (float(bytes_count) / bandwidth_bytes_per_sec) * 1000.0


def _max_abs_error(expected: np.ndarray | None, actual: np.ndarray) -> float | None:
    if expected is None:
        return None
    if expected.shape != actual.shape:
        raise ValueError(f"expected KV shape {expected.shape} does not match restored shape {actual.shape}")
    difference = np.subtract(expected, actual, dtype=np.float32)
    np.abs(difference, out=difference)
    return float(np.max(difference))


def _decode_restore_artifact(artifact_dir: Path, *, variant_name: str | None) -> tuple[np.ndarray, int, float, float]:
    manifest = read_manifest(artifact_dir / "manifest.json")
    batches = []
    decode_ms = 0.0
    network_bytes = 0
    for part in select_manifest_parts(manifest, variant_name=variant_name):
        data = verify_part_payload(artifact_dir, part)
        network_bytes += part.payload_bytes

        decode_start = perf_counter()
        batches.append(decode_part_payload(manifest, part, data, variant_name=variant_name))
        decode_ms += _elapsed_ms(decode_start)

    from cradle_codec.pipeline import layout_from_manifest

    layout = layout_from_manifest(manifest, variant_name=variant_name)
    restore_start = perf_counter()
    restored = unpack_frame_batches_to_kv(batches, layout, num_tokens=manifest.kv_shape.num_tokens)
    restore_ms = _elapsed_ms(restore_start)
    return restored, network_bytes, decode_ms, restore_ms


def benchmark_artifact_reuse(
    artifact_dir: str | Path,
    *,
    expected_kv: np.ndarray | None = None,
    bandwidth_bytes_per_sec: float,
    prefill_ms: float,
    scheduler_wait_ms: float = 0.0,
    variant_name: str | None = None,
) -> BenchmarkReport:
    """Measure deterministic local pieces of a remote KV reuse decision.

    The report separates what this library can observe (artifact bytes,
    decode/dequantize time, restoration time, accuracy) from simulated serving
    inputs (network bandwidth, full-prefill baseline, scheduler wait). The TTFT
    estimates follow the paper's high-level comparison: full prefill, raw remote
    KV transfer, and codec-backed remote KV transfer plus decode/restore.
    """

    artifact_dir = Path(artifact_dir)
    bandwidth = _positive_float("bandwidth_bytes_per_sec", bandwidth_bytes_per_sec)
    prefill_ms = _positive_float("prefill_ms", prefill_ms)
    scheduler_wait_ms = float(scheduler_wait_ms)
    if scheduler_wait_ms < 0.0:
        raise ValueError("scheduler_wait_ms must be non-negative")

    manifest = read_manifest(artifact_dir / "manifest.json")
    if expected_kv is not None:
        raw_bytes = int(expected_kv.nbytes)
    else:
        source_dtype = manifest.parts[0].quantization.source_dtype if manifest.parts else "float32"
        raw_bytes = (
            manifest.kv_shape.num_sides
            * manifest.kv_shape.num_layers
            * manifest.kv_shape.num_tokens
            * manifest.kv_shape.num_kv_heads
            * manifest.kv_shape.head_dim
            * np.dtype(source_dtype).itemsize
        )

    selected_variant = variant_name
    if selected_variant is None:
        selected_variant = select_variant(manifest, bandwidth_bytes_per_sec=bandwidth).variant.name

    restored, codec_bytes, decode_ms, restore_ms = _decode_restore_artifact(artifact_dir, variant_name=selected_variant)
    max_abs_error = _max_abs_error(expected_kv, restored)
    encoded_to_raw = codec_bytes / raw_bytes if raw_bytes else 0.0
    raw_to_encoded = raw_bytes / codec_bytes if codec_bytes else float("inf")

    full_prefill = BenchmarkMetric(
        method="full_prefill",
        ttft_ms=prefill_ms,
        network_bytes=0,
        decode_ms=0.0,
        restore_ms=0.0,
        scheduler_wait_ms=0.0,
    )
    raw_kv = BenchmarkMetric(
        method="raw_kv_reuse",
        ttft_ms=scheduler_wait_ms + _network_ms(raw_bytes, bandwidth),
        network_bytes=raw_bytes,
        decode_ms=0.0,
        restore_ms=0.0,
        scheduler_wait_ms=scheduler_wait_ms,
        encoded_to_raw_ratio=1.0,
        raw_to_encoded_ratio=1.0,
        max_abs_error=0.0 if expected_kv is not None else None,
    )
    codec_reuse = BenchmarkMetric(
        method="codec_reuse",
        ttft_ms=scheduler_wait_ms + _network_ms(codec_bytes, bandwidth) + decode_ms + restore_ms,
        network_bytes=codec_bytes,
        decode_ms=decode_ms,
        restore_ms=restore_ms,
        scheduler_wait_ms=scheduler_wait_ms,
        encoded_to_raw_ratio=encoded_to_raw,
        raw_to_encoded_ratio=raw_to_encoded,
        max_abs_error=max_abs_error,
        selected_variant=selected_variant,
    )
    return BenchmarkReport(
        source_key=manifest.source_key,
        model=manifest.model,
        raw_bytes=raw_bytes,
        bandwidth_bytes_per_sec=bandwidth,
        metrics=(full_prefill, raw_kv, codec_reuse),
    )


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    return {
        "source_key": report.source_key,
        "model": report.model,
        "raw_bytes": report.raw_bytes,
        "bandwidth_bytes_per_sec": report.bandwidth_bytes_per_sec,
        "metrics": [asdict(metric) for metric in report.metrics],
    }


class LiveBenchmarkError(RuntimeError):
    """Base class for real-model benchmark failures with actionable messages."""


class LiveBenchmarkDependencyError(LiveBenchmarkError):
    """Raised when an optional serving backend is not installed."""


class LiveBenchmarkRuntimeError(LiveBenchmarkError):
    """Raised when a backend is installed but cannot run the requested model."""


@dataclass(frozen=True)
class LiveModelRequestMetric:
    request_index: int
    request_id: str
    prompt_sha256: str
    prompt_preview: str
    prompt_chars: int
    prompt_tokens: int | None
    generated_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    total_ms: float
    output_tokens_per_second: float | None


@dataclass(frozen=True)
class LiveModelBenchmarkReport:
    backend: Literal["transformers", "vllm"]
    model: str
    device: str
    dtype: str
    trust_remote_code: bool
    max_new_tokens: int
    sampling: dict[str, Any]
    prompt_count: int
    started_at: str
    ended_at: str
    total_ms: float
    metrics: tuple[LiveModelRequestMetric, ...]

    def to_json(self) -> str:
        return json.dumps(live_model_report_to_dict(self), sort_keys=True, indent=2)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_module(import_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        raise LiveBenchmarkDependencyError(
            f"optional dependency {import_name!r} is required for this live benchmark backend; "
            f"install it with {install_hint}"
        )


def _normalize_backend(backend: str) -> Literal["transformers", "vllm"]:
    normalized = backend.strip().lower()
    if normalized not in {"transformers", "vllm"}:
        raise ValueError("backend must be one of: transformers, vllm")
    return normalized  # type: ignore[return-value]


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _sampling_dict(*, temperature: float, top_p: float, seed: int | None) -> dict[str, Any]:
    return {"temperature": float(temperature), "top_p": float(top_p), "seed": seed}


def _maybe_seed_torch(torch_module: Any, seed: int | None) -> None:
    if seed is None:
        return
    torch_module.manual_seed(int(seed))
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        cuda.manual_seed_all(int(seed))


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    normalized = dtype.strip().removeprefix("torch.")
    if normalized == "auto":
        return "auto"
    if not hasattr(torch_module, normalized):
        raise ValueError(f"unknown torch dtype {dtype!r}; use auto, float16, bfloat16, or float32")
    return getattr(torch_module, normalized)


def _first_parameter_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def _move_inputs(inputs: dict[str, Any], device: Any | None) -> dict[str, Any]:
    if device is None:
        return inputs
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}


def _supported_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if value is not None and key in signature.parameters}


def _prompt_preview(prompt: str) -> str:
    return prompt[:157] + "..." if len(prompt) > 160 else prompt


def _live_metric(
    *,
    request_index: int,
    request_id: str,
    prompt: str,
    prompt_tokens: int | None,
    token_times: list[float],
    start: float,
    end: float,
) -> LiveModelRequestMetric:
    generated_tokens = len(token_times)
    total_ms = (end - start) * 1000.0
    ttft_ms = (token_times[0] - start) * 1000.0 if token_times else None
    tpot_ms = ((token_times[-1] - token_times[0]) * 1000.0 / (generated_tokens - 1)) if generated_tokens > 1 else None
    output_tokens_per_second = (generated_tokens / (total_ms / 1000.0)) if total_ms > 0.0 else None
    return LiveModelRequestMetric(
        request_index=request_index,
        request_id=request_id,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_preview=_prompt_preview(prompt),
        prompt_chars=len(prompt),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        output_tokens_per_second=output_tokens_per_second,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _live_aggregate(metrics: tuple[LiveModelRequestMetric, ...]) -> dict[str, Any]:
    ttft_values = [metric.ttft_ms for metric in metrics if metric.ttft_ms is not None]
    tpot_values = [metric.tpot_ms for metric in metrics if metric.tpot_ms is not None]
    total_values = [metric.total_ms for metric in metrics]
    generated_tokens = sum(metric.generated_tokens for metric in metrics)
    total_seconds = sum(total_values) / 1000.0
    return {
        "requests_completed": len(metrics),
        "generated_tokens": generated_tokens,
        "ttft_ms_mean": _mean(ttft_values),
        "ttft_ms_p50": _percentile(ttft_values, 0.50),
        "ttft_ms_p95": _percentile(ttft_values, 0.95),
        "tpot_ms_mean": _mean(tpot_values),
        "tpot_ms_p50": _percentile(tpot_values, 0.50),
        "tpot_ms_p95": _percentile(tpot_values, 0.95),
        "total_ms_mean": _mean(total_values),
        "aggregate_output_tokens_per_second": (generated_tokens / total_seconds) if total_seconds > 0.0 else None,
    }


class _GenerationTimingStreamer:
    """Minimal transformers-compatible streamer that records generated token arrival times."""

    def __init__(self, prompt_tokens: int) -> None:
        self._remaining_prompt_tokens = prompt_tokens
        self.token_times: list[float] = []
        self._done = threading.Event()
        self._errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def put(self, value: Any) -> None:
        try:
            tokens = value.tolist() if hasattr(value, "tolist") else value
            if not isinstance(tokens, list):
                tokens = [tokens]
            while tokens and isinstance(tokens[0], list):
                if len(tokens) != 1:
                    raise ValueError("live transformers benchmark expects batch size 1")
                tokens = tokens[0]
            if self._remaining_prompt_tokens:
                skipped = min(self._remaining_prompt_tokens, len(tokens))
                self._remaining_prompt_tokens -= skipped
                tokens = tokens[skipped:]
            if tokens:
                now = perf_counter()
                self.token_times.extend(now for _ in tokens)
        except BaseException as exc:
            self._errors.put(exc)
            self.end()

    def end(self) -> None:
        self._done.set()

    def put_error(self, exc: BaseException) -> None:
        self._errors.put(exc)
        self.end()

    def wait(self) -> None:
        self._done.wait()
        if not self._errors.empty():
            raise self._errors.get()


def _run_transformers_live_model_benchmark(
    *,
    model: str,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dtype: str,
    device: str,
    device_map: str | None,
    trust_remote_code: bool,
    seed: int | None,
) -> tuple[LiveModelRequestMetric, ...]:
    _require_module("torch", "pip install torch")
    _require_module("transformers", "pip install transformers accelerate")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise LiveBenchmarkRuntimeError("CUDA was requested for transformers but torch.cuda.is_available() is false")

    _maybe_seed_torch(torch, seed)
    resolved_dtype = _resolve_torch_dtype(torch, dtype)
    normalized_device_map = None if device_map is None or device_map.lower() in {"", "none"} else device_map
    if normalized_device_map is not None:
        _require_module("accelerate", "pip install accelerate")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {"torch_dtype": resolved_dtype, "trust_remote_code": trust_remote_code}
        if normalized_device_map is not None:
            model_kwargs["device_map"] = normalized_device_map
        loaded_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        loaded_model.eval()
        if normalized_device_map is None:
            loaded_model.to(torch.device(device))
    except Exception as exc:
        raise LiveBenchmarkRuntimeError(f"failed to load {model!r} with transformers: {exc}") from exc

    input_device = _first_parameter_device(loaded_model)
    if normalized_device_map is None:
        input_device = torch.device(device)

    metrics: list[LiveModelRequestMetric] = []
    for index, prompt in enumerate(prompts):
        try:
            encoded = tokenizer(prompt, return_tensors="pt")
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            encoded = _move_inputs(dict(encoded), input_device)
            streamer = _GenerationTimingStreamer(prompt_tokens)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "top_p": top_p,
                "streamer": streamer,
            }
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
            if getattr(tokenizer, "eos_token_id", None) is not None:
                generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

            def generate() -> None:
                try:
                    with torch.inference_mode():
                        loaded_model.generate(**encoded, **generate_kwargs)
                except BaseException as exc:
                    streamer.put_error(exc)
                finally:
                    streamer.end()

            request_id = f"transformers-{uuid.uuid4()}"
            start = perf_counter()
            worker = threading.Thread(target=generate, name=f"kvcodec-{request_id}", daemon=True)
            worker.start()
            streamer.wait()
            worker.join()
            end = perf_counter()
            metrics.append(
                _live_metric(
                    request_index=index,
                    request_id=request_id,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    token_times=streamer.token_times,
                    start=start,
                    end=end,
                )
            )
        except Exception as exc:
            raise LiveBenchmarkRuntimeError(f"transformers generation failed for prompt index {index}: {exc}") from exc
    return tuple(metrics)


async def _run_vllm_live_model_benchmark_async(
    *,
    model: str,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dtype: str,
    trust_remote_code: bool,
    seed: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
) -> tuple[LiveModelRequestMetric, ...]:
    _require_module("torch", "pip install torch")
    _require_module("vllm", "pip install vllm")
    import torch
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    if not torch.cuda.is_available():
        raise LiveBenchmarkRuntimeError("vLLM backend requires CUDA, but torch.cuda.is_available() is false")
    _maybe_seed_torch(torch, seed)

    try:
        engine_args = AsyncEngineArgs(
            **_supported_kwargs(
                AsyncEngineArgs,
                {
                    "model": model,
                    "dtype": dtype,
                    "trust_remote_code": trust_remote_code,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "max_model_len": max_model_len,
                    "disable_log_stats": True,
                },
            )
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    except Exception as exc:
        raise LiveBenchmarkRuntimeError(f"failed to initialize vLLM engine for {model!r}: {exc}") from exc

    metrics: list[LiveModelRequestMetric] = []
    try:
        sampling_params = SamplingParams(
            **_supported_kwargs(
                SamplingParams,
                {
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                },
            )
        )
        for index, prompt in enumerate(prompts):
            request_id = f"vllm-{uuid.uuid4()}"
            token_times: list[float] = []
            seen_tokens = 0
            prompt_tokens: int | None = None
            start = perf_counter()
            try:
                async for request_output in engine.generate(prompt, sampling_params, request_id):
                    now = perf_counter()
                    output_prompt_tokens = getattr(request_output, "prompt_token_ids", None)
                    if output_prompt_tokens is not None:
                        prompt_tokens = len(output_prompt_tokens)
                    outputs = getattr(request_output, "outputs", None) or []
                    if not outputs:
                        continue
                    token_ids = getattr(outputs[0], "token_ids", None) or []
                    current_tokens = len(token_ids)
                    if current_tokens > seen_tokens:
                        token_times.extend(now for _ in range(current_tokens - seen_tokens))
                        seen_tokens = current_tokens
            except Exception as exc:
                raise LiveBenchmarkRuntimeError(f"vLLM generation failed for prompt index {index}: {exc}") from exc
            end = perf_counter()
            metrics.append(
                _live_metric(
                    request_index=index,
                    request_id=request_id,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    token_times=token_times,
                    start=start,
                    end=end,
                )
            )
    finally:
        shutdown = getattr(engine, "shutdown_background_loop", None)
        if callable(shutdown):
            shutdown()
    return tuple(metrics)


def run_live_model_benchmark(
    *,
    backend: str,
    model: str = "Qwen/Qwen3-1.7B",
    prompts: tuple[str, ...],
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    dtype: str = "auto",
    device: str = "cuda",
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    seed: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
) -> LiveModelBenchmarkReport:
    """Run a dependency-light real-model TTFT/TPOT benchmark.

    Heavy serving libraries are imported only inside the selected backend path.
    The function raises LiveBenchmarkDependencyError when optional packages are
    missing and LiveBenchmarkRuntimeError when the model, CUDA runtime, or
    backend engine cannot be initialized.
    """

    normalized_backend = _normalize_backend(backend)
    prompts = tuple(prompt for prompt in prompts if prompt)
    if not prompts:
        raise ValueError("at least one non-empty prompt is required")
    max_new_tokens = _positive_int("max_new_tokens", max_new_tokens)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    started_at = _utc_timestamp()
    total_start = perf_counter()
    if normalized_backend == "transformers":
        metrics = _run_transformers_live_model_benchmark(
            model=model,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dtype=dtype,
            device=device,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    else:
        metrics = asyncio.run(
            _run_vllm_live_model_benchmark_async(
                model=model,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        )
    total_ms = _elapsed_ms(total_start)
    ended_at = _utc_timestamp()
    return LiveModelBenchmarkReport(
        backend=normalized_backend,
        model=model,
        device=device if normalized_backend == "transformers" else "cuda",
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        sampling=_sampling_dict(temperature=temperature, top_p=top_p, seed=seed),
        prompt_count=len(prompts),
        started_at=started_at,
        ended_at=ended_at,
        total_ms=total_ms,
        metrics=metrics,
    )


def live_model_report_to_dict(report: LiveModelBenchmarkReport) -> dict[str, Any]:
    return {
        "backend": report.backend,
        "model": report.model,
        "device": report.device,
        "dtype": report.dtype,
        "trust_remote_code": report.trust_remote_code,
        "max_new_tokens": report.max_new_tokens,
        "sampling": dict(report.sampling),
        "prompt_count": report.prompt_count,
        "started_at": report.started_at,
        "ended_at": report.ended_at,
        "total_ms": report.total_ms,
        "aggregate": _live_aggregate(report.metrics),
        "metrics": [asdict(metric) for metric in report.metrics],
    }
