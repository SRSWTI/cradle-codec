from __future__ import annotations

import json
import os
import importlib.util
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from cradle_codec.integration import (
    KVCodecLMCacheStoragePluginConfig,
    QWEN3_17B_MODEL,
    lmcache_inprocess_kv_transfer_config,
    lmcache_kvcodec_storage_env,
    lmcache_mp_kv_transfer_config,
    vllm_serve_args,
)
from cradle_codec.layout import HeadDimTiling, KVCodecLayout, candidate_name_for_tiling, select_layout_candidate
from cradle_codec.pipeline import decode_kv_artifact, encode_kv_chunk
from cradle_codec.quant import QuantizationSpec, compute_error_metrics


@dataclass(frozen=True)
class StreamRequestTiming:
    name: str
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    chunk_count: int
    text: str
    usage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class VLLMLMCacheStackReport:
    model: str
    mode: str
    vllm_base_url: str
    lmcache_zmq_url: str | None
    request_timings: tuple[StreamRequestTiming, ...]
    log_dir: str

    def to_json(self) -> str:
        return json.dumps(vllm_lmcache_report_to_dict(self), sort_keys=True, indent=2)


@dataclass(frozen=True)
class TransformersKVCodecReport:
    model: str
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    ttft_ms: float
    tpot_ms: float
    kv_shape: tuple[int, int, int, int, int]
    layout_name: str
    artifact_dir: str
    raw_bytes: int
    encoded_bytes: int
    encoded_to_raw_ratio: float
    raw_to_encoded_ratio: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    cosine_similarity: float | None

    def to_json(self) -> str:
        return json.dumps(transformers_kvcodec_report_to_dict(self), sort_keys=True, indent=2)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required executable {name!r} is not on PATH; install the serving optional dependencies")
    return resolved


def _post_json_stream(url: str, payload: Mapping[str, Any], *, timeout_s: float) -> Iterable[dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - caller controls local benchmark URL
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                yield "[DONE]"
                return
            yield json.loads(line)


def time_openai_completion_stream(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_name: str,
    timeout_s: float = 300.0,
) -> StreamRequestTiming:
    """Send one OpenAI-compatible streaming completion and measure TTFT/TPOT."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    text_parts: list[str] = []
    usage: Mapping[str, Any] | None = None
    for event in _post_json_stream(url, payload, timeout_s=timeout_s):
        if event == "[DONE]":
            break
        choices = event.get("choices") or []
        emitted_text = ""
        if choices:
            emitted_text = str(choices[0].get("text") or "")
        if emitted_text:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks += 1
            text_parts.append(emitted_text)
        if event.get("usage") is not None:
            usage = event["usage"]
    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream completed without emitting any token text")
    total_ms = (end - start) * 1000.0
    ttft_ms = (first_token_at - start) * 1000.0
    tpot_ms = 0.0 if chunks <= 1 else ((end - first_token_at) * 1000.0) / float(chunks - 1)
    return StreamRequestTiming(
        name=request_name,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_ms=total_ms,
        chunk_count=chunks,
        text="".join(text_parts),
        usage=usage,
    )


def _read_log_tail(path: Path, *, max_bytes: int = 8192) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"


def _assert_process_running(process: subprocess.Popen[bytes], *, name: str, log_path: Path) -> None:
    return_code = process.poll()
    if return_code is None:
        return
    tail = _read_log_tail(log_path)
    raise RuntimeError(f"{name} exited before becoming ready with code {return_code}. Log tail:\n{tail}")


def _assert_tcp_port_free(*, host: str, port: int, name: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"{name} port {port} is not available on {host}: {exc}") from exc


def _wait_for_http_ok(
    url: str,
    *,
    timeout_s: float,
    interval_s: float = 1.0,
    process: subprocess.Popen[bytes] | None = None,
    process_name: str | None = None,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process_name is not None and log_path is not None:
            _assert_process_running(process, name=process_name, log_path=log_path)
        try:
            with urlopen(url, timeout=min(interval_s, 5.0)) as response:  # noqa: S310 - local benchmark URL
                if 200 <= int(response.status) < 300:
                    return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(interval_s)
    if process is not None and process_name is not None and log_path is not None:
        _assert_process_running(process, name=process_name, log_path=log_path)
    raise TimeoutError(f"timed out waiting for {url}; last error: {last_error}")


def _terminate_process(process: subprocess.Popen[bytes], *, grace_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=grace_s)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
    if process.poll() is None:
        process.kill()
        process.wait(timeout=grace_s)


def _require_optional_module(import_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"optional dependency {import_name!r} is required; install it with {install_hint}")


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    normalized = dtype.strip().removeprefix("torch.")
    if normalized == "auto":
        return "auto"
    if normalized not in {"float16", "bfloat16", "float32"} or not hasattr(torch_module, normalized):
        raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")
    return getattr(torch_module, normalized)


def _first_parameter_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def run_vllm_lmcache_live_benchmark(
    *,
    model: str = QWEN3_17B_MODEL,
    prompt: str,
    max_tokens: int = 32,
    requests: int = 2,
    mode: str = "mp",
    vllm_port: int = 8000,
    lmcache_port: int = 5555,
    lmcache_http_port: int = 18080,
    lmcache_l1_gb: float = 4.0,
    startup_timeout_s: float = 900.0,
    request_timeout_s: float = 300.0,
    log_dir: str | Path = "live-vllm-lmcache-logs",
    use_lmcache_shipped_connector: bool = True,
    extra_vllm_args: Iterable[str] = (),
    kvcodec_storage_plugin: KVCodecLMCacheStoragePluginConfig | None = None,
) -> VLLMLMCacheStackReport:
    """Launch vLLM with LMCache, then measure live streaming TTFT/TPOT traffic.

    ``mode="mp"`` starts the standalone LMCache server and exercises the MP
    connector. ``mode="in-process"`` uses vLLM's LMCacheConnectorV1, which is
    the path where LMCache ``storage_plugins`` are loaded from
    ``LMCacheEngineConfig`` in the vLLM process.
    """

    if requests <= 0:
        raise ValueError("requests must be positive")
    if lmcache_l1_gb <= 0:
        raise ValueError("lmcache_l1_gb must be positive")
    normalized_mode = mode.strip().lower().replace("_", "-")
    if normalized_mode not in {"mp", "in-process"}:
        raise ValueError("mode must be 'mp' or 'in-process'")

    _require_executable("vllm")
    if normalized_mode == "mp":
        _require_executable("lmcache")

    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("LMCACHE_DISABLE_BANNER", "1")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    if kvcodec_storage_plugin is not None:
        env = lmcache_kvcodec_storage_env(kvcodec_storage_plugin, base_env=env)

    _assert_tcp_port_free(host="0.0.0.0", port=vllm_port, name="vLLM HTTP")
    if normalized_mode == "mp":
        _assert_tcp_port_free(host="127.0.0.1", port=lmcache_port, name="LMCache ZMQ")
        _assert_tcp_port_free(host="0.0.0.0", port=lmcache_http_port, name="LMCache HTTP")

    vllm_stdout = (logs / "vllm.out.log").open("wb")
    lmcache_stdout: Any | None = None
    lmcache_proc: subprocess.Popen[bytes] | None = None
    vllm_proc: subprocess.Popen[bytes] | None = None
    try:
        if normalized_mode == "mp":
            lmcache_stdout = (logs / "lmcache.out.log").open("wb")
            lmcache_cmd = [
                "lmcache",
                "server",
                "--host",
                "localhost",
                "--port",
                str(lmcache_port),
                "--http-port",
                str(lmcache_http_port),
                "--l1-size-gb",
                str(lmcache_l1_gb),
                "--eviction-policy",
                "LRU",
                "--disable-logging",
            ]
            lmcache_proc = subprocess.Popen(lmcache_cmd, stdout=lmcache_stdout, stderr=subprocess.STDOUT, env=env)
            _wait_for_http_ok(
                f"http://127.0.0.1:{lmcache_http_port}/healthcheck",
                timeout_s=120.0,
                process=lmcache_proc,
                process_name="LMCache server",
                log_path=logs / "lmcache.out.log",
            )
            kv_config = lmcache_mp_kv_transfer_config(
                host="tcp://localhost",
                port=lmcache_port,
                use_lmcache_shipped_connector=use_lmcache_shipped_connector,
            )
            lmcache_zmq_url = f"tcp://localhost:{lmcache_port}"
        else:
            kv_config = lmcache_inprocess_kv_transfer_config()
            lmcache_zmq_url = None

        vllm_args = vllm_serve_args(
            model=model,
            port=vllm_port,
            kv_transfer_config=kv_config,
            extra_args=tuple(extra_vllm_args),
        )
        vllm_proc = subprocess.Popen(list(vllm_args), stdout=vllm_stdout, stderr=subprocess.STDOUT, env=env)
        base_url = f"http://127.0.0.1:{vllm_port}"
        _wait_for_http_ok(
            base_url + "/health",
            timeout_s=startup_timeout_s,
            process=vllm_proc,
            process_name="vLLM server",
            log_path=logs / "vllm.out.log",
        )

        timings = []
        for index in range(requests):
            timings.append(
                time_openai_completion_stream(
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    request_name=f"request_{index + 1}",
                    timeout_s=request_timeout_s,
                )
            )
        return VLLMLMCacheStackReport(
            model=model,
            mode=normalized_mode,
            vllm_base_url=base_url,
            lmcache_zmq_url=lmcache_zmq_url,
            request_timings=tuple(timings),
            log_dir=str(logs),
        )
    finally:
        if vllm_proc is not None:
            _terminate_process(vllm_proc)
        if lmcache_proc is not None:
            _terminate_process(lmcache_proc)
        vllm_stdout.close()
        if lmcache_stdout is not None:
            lmcache_stdout.close()


def _legacy_past_layers(past_key_values: Any) -> list[tuple[Any, Any]]:
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache, strict=True))
    layers = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise ValueError(f"cannot extract key/value tensors from cache layer of type {type(layer).__name__}")
        layers.append((layer[0], layer[1]))
    return layers


def _layer_tensor_to_token_head_dim(tensor: Any, *, token_count: int) -> Any:
    # Expected common forms are [batch, heads, tokens, dim] and [batch, tokens, heads, dim].
    if tensor.ndim != 4:
        raise ValueError(f"expected layer KV tensor rank 4, got {tensor.ndim}")
    if int(tensor.shape[0]) != 1:
        raise ValueError(f"only batch size 1 is supported for KV extraction, got batch={int(tensor.shape[0])}")
    no_batch = tensor[0]
    if int(no_batch.shape[1]) == token_count:
        return no_batch.permute(1, 0, 2).contiguous()
    if int(no_batch.shape[0]) == token_count:
        return no_batch.contiguous()
    raise ValueError(f"cannot identify token axis in KV tensor shape {tuple(tensor.shape)} for token_count={token_count}")


def canonical_kv_from_past_key_values(past_key_values: Any, *, token_count: int) -> np.ndarray:
    """Convert Transformers past_key_values to canonical [2,L,T,H,D] float32."""

    layers = _legacy_past_layers(past_key_values)
    if not layers:
        raise ValueError("past_key_values did not contain any layers")
    k_layers = []
    v_layers = []
    for key_tensor, value_tensor in layers:
        k_layers.append(_layer_tensor_to_token_head_dim(key_tensor.detach().cpu(), token_count=token_count).numpy())
        v_layers.append(_layer_tensor_to_token_head_dim(value_tensor.detach().cpu(), token_count=token_count).numpy())
    k = np.stack(k_layers, axis=0)
    v = np.stack(v_layers, axis=0)
    return np.stack([k, v], axis=0).astype(np.float32, copy=False)


def _default_layout_for_kv(kv: np.ndarray, *, layout_name: str | None, layers_per_frame: int) -> KVCodecLayout:
    if kv.ndim != 5:
        raise ValueError(f"expected canonical KV rank 5, got {kv.ndim}")
    num_layers = int(kv.shape[1])
    num_kv_heads = int(kv.shape[3])
    head_dim = int(kv.shape[4])
    if layout_name:
        from cradle_codec.layout import layout_from_name

        return layout_from_name(layout_name, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, layers_per_frame=layers_per_frame)
    profile = select_layout_candidate(kv, layers_per_frame=layers_per_frame)
    return profile.layout


def _codec_from_name(name: str, *, nvenc_workers: int = 1, nvdec_workers: int = 1) -> FrameCodec:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"pynv", "pynvvideocodec", "pynv-hevc", "nvenc"}:
        from cradle_codec.codec import PyNvVideoCodecHEVCCodec

        return PyNvVideoCodecHEVCCodec(nvenc_workers=nvenc_workers, nvdec_workers=nvdec_workers)
    if normalized in {"ffmpeg", "ffmpeg-hevc", "libx265"}:
        from cradle_codec.codec import FFmpegHEVCCodec

        return FFmpegHEVCCodec()
    if normalized in {"reference", "raw-reference", "raw"}:
        from cradle_codec.codec import RawReferenceCodec

        return RawReferenceCodec()
    raise ValueError("codec must be one of: pynvvideocodec, ffmpeg, reference")


def run_transformers_kvcodec_artifact_benchmark(
    *,
    model: str = QWEN3_17B_MODEL,
    prompt: str,
    max_new_tokens: int = 16,
    artifact_dir: str | Path,
    layout_name: str | None = None,
    layers_per_frame: int = 3,
    codec_name: str = "pynvvideocodec",
    quant_axis: str = "channel",
    nvenc_workers: int = 1,
    nvdec_workers: int = 1,
    device_map: str | None = "auto",
    dtype: str = "auto",
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> TransformersKVCodecReport:
    """Run one real Transformers inference, encode its KV cache, decode it, and report error."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    _require_optional_module("torch", "uv add --optional serving torch")
    _require_optional_module("transformers", "uv add --optional serving transformers")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = _resolve_torch_dtype(torch, dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": trust_remote_code}
        if device_map and device_map != "none":
            model_kwargs["device_map"] = device_map
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("transformers backend requires CUDA for device='cuda', but torch.cuda.is_available() is false")
        torch_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        if not device_map or device_map == "none":
            torch_model = torch_model.to(torch.device(device))
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - depends on network/model/runtime state
        raise RuntimeError(f"failed to load Transformers model {model!r}: {exc}") from exc
    torch_model.eval()
    input_device = _first_parameter_device(torch_model) or torch.device(device)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    with torch.inference_mode():
        prefill_start = time.perf_counter()
        outputs = torch_model(**encoded, use_cache=True)
        prefill_ms = _elapsed_ms(prefill_start)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ttft_ms = prefill_ms
        generated = [int(next_token.item())]
        past = past_key_values
        decode_durations = []
        for _ in range(max_new_tokens - 1):
            decode_start = time.perf_counter()
            step = torch_model(input_ids=next_token, past_key_values=past, use_cache=True)
            decode_durations.append(_elapsed_ms(decode_start))
            past = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
    tpot_ms = 0.0 if not decode_durations else float(sum(decode_durations) / len(decode_durations))

    kv = canonical_kv_from_past_key_values(past_key_values, token_count=prompt_tokens)
    layout = _default_layout_for_kv(kv, layout_name=layout_name, layers_per_frame=layers_per_frame)
    artifact_path = Path(artifact_dir)
    manifest = encode_kv_chunk(
        kv,
        artifact_path,
        source_key=f"transformers:{model}#prompt_tokens={prompt_tokens}",
        model=model,
        layout=layout,
        quantization=QuantizationSpec(mode="uint8_minmax", axis=quant_axis),  # type: ignore[arg-type]
        codec=_codec_from_name(codec_name, nvenc_workers=nvenc_workers, nvdec_workers=nvdec_workers),
    )
    restored = decode_kv_artifact(artifact_path)
    metrics = compute_error_metrics(kv, restored)
    encoded_bytes = sum(part.payload_bytes for part in manifest.sorted_parts())
    raw_bytes = int(kv.nbytes)
    return TransformersKVCodecReport(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated),
        prefill_ms=prefill_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        kv_shape=tuple(int(dim) for dim in kv.shape),
        layout_name=candidate_name_for_tiling(layout.tiling),
        artifact_dir=str(artifact_path),
        raw_bytes=raw_bytes,
        encoded_bytes=encoded_bytes,
        encoded_to_raw_ratio=encoded_bytes / raw_bytes if raw_bytes else 0.0,
        raw_to_encoded_ratio=raw_bytes / encoded_bytes if encoded_bytes else float("inf"),
        max_abs_error=metrics.max_abs_error,
        mean_abs_error=metrics.mean_abs_error,
        rmse=metrics.rmse,
        cosine_similarity=metrics.cosine_similarity,
    )


def vllm_lmcache_report_to_dict(report: VLLMLMCacheStackReport) -> dict[str, Any]:
    return {
        "model": report.model,
        "mode": report.mode,
        "vllm_base_url": report.vllm_base_url,
        "lmcache_zmq_url": report.lmcache_zmq_url,
        "log_dir": report.log_dir,
        "request_timings": [asdict(timing) for timing in report.request_timings],
    }


def transformers_kvcodec_report_to_dict(report: TransformersKVCodecReport) -> dict[str, Any]:
    return asdict(report)
