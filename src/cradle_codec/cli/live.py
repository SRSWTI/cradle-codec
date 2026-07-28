from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer

from cradle_codec.benchmark import LiveBenchmarkError, run_live_model_benchmark
from cradle_codec.integration import QWEN3_17B_MODEL, lmcache_kvcodec_storage_plugin_config
from cradle_codec.live import run_transformers_kvcodec_artifact_benchmark, run_vllm_lmcache_live_benchmark

app = typer.Typer(help="Run real-model TTFT/TPOT and KVCodec artifact benchmarks.", no_args_is_help=True)


def _load_prompts(prompt: tuple[str, ...], prompt_file: Optional[Path]) -> tuple[str, ...]:
    prompts = [item for item in prompt if item]
    if prompt_file is not None:
        prompts.extend(line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not prompts:
        raise typer.BadParameter("provide at least one --prompt or a non-empty --prompt-file")
    return tuple(prompts)


@app.command("model")
def model_command(
    backend: str = typer.Option("transformers", "--backend", help="Backend: transformers or vllm."),
    model: str = typer.Option(QWEN3_17B_MODEL, "--model", help="Hugging Face model id."),
    prompt: list[str] = typer.Option(["KVCodec live model smoke prompt: explain cache reuse in one sentence."], "--prompt", help="Prompt to run; repeat for multiple requests."),
    prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", exists=True, dir_okay=False, file_okay=True, help="UTF-8 file with one non-empty prompt per line."),
    max_new_tokens: int = typer.Option(16, "--max-new-tokens", min=1, help="Generated tokens per request."),
    temperature: float = typer.Option(0.0, "--temperature", min=0.0, help="Sampling temperature; 0 uses greedy decoding."),
    top_p: float = typer.Option(1.0, "--top-p", min=0.0, max=1.0, help="Nucleus sampling top-p."),
    dtype: str = typer.Option("auto", "--dtype", help="Model dtype: auto, float16, bfloat16, or float32."),
    device: str = typer.Option("cuda", "--device", help="Transformers device when --device-map none is used."),
    device_map: Optional[str] = typer.Option("auto", "--device-map", help="Transformers device_map; use none to disable."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Allow custom model code from Hugging Face."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Optional random seed."),
    gpu_memory_utilization: Optional[float] = typer.Option(None, "--gpu-memory-utilization", min=0.0, max=1.0, help="vLLM GPU memory fraction."),
    max_model_len: Optional[int] = typer.Option(None, "--max-model-len", min=1, help="vLLM maximum model context length."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional path to write the JSON report."),
) -> None:
    """Run live TTFT/TPOT generation with a real model backend."""

    try:
        report = run_live_model_benchmark(
            backend=backend,
            model=model,
            prompts=_load_prompts(tuple(prompt), prompt_file),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dtype=dtype,
            device=device,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    except LiveBenchmarkError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    payload = report.to_json()
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload + "\n", encoding="utf-8")
    typer.echo(payload)


@app.command("transformers-kv")
def transformers_kv_command(
    model: str = typer.Option(QWEN3_17B_MODEL, "--model", help="Hugging Face model id."),
    prompt: str = typer.Option("KVCodec real-model smoke prompt: explain cache reuse in one sentence.", "--prompt", help="Prompt used for one real inference."),
    max_new_tokens: int = typer.Option(8, "--max-new-tokens", min=1, help="Greedy decode tokens used for TPOT timing."),
    artifact_dir: Path = typer.Option(..., "--artifact", "-a", file_okay=False, dir_okay=True, help="Artifact output directory."),
    layout_name: Optional[str] = typer.Option(None, "--layout", help="Optional paper-style layout name, e.g. h1x8_d32x4_32x32."),
    layers_per_frame: int = typer.Option(3, "--layers-per-frame", min=1, max=3, help="Adjacent layers packed into frame channels."),
    codec: str = typer.Option("pynvvideocodec", "--codec", help="Frame codec: pynvvideocodec, ffmpeg, or reference."),
    quant_axis: str = typer.Option("channel", "--quant-axis", help="uint8_minmax axis: part, frame, or channel."),
    nvenc_workers: int = typer.Option(1, "--nvenc-workers", min=1, help="PyNvVideoCodec NVENC worker count."),
    nvdec_workers: int = typer.Option(1, "--nvdec-workers", min=1, help="PyNvVideoCodec NVDEC worker count recorded for artifact decode."),
    dtype: str = typer.Option("auto", "--dtype", help="Transformers dtype: auto, float16, bfloat16, or float32."),
    device: str = typer.Option("cuda", "--device", help="Transformers device when --device-map none is used."),
    device_map: Optional[str] = typer.Option("auto", "--device-map", help="Transformers device_map; use none to disable."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Allow custom model code from Hugging Face."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional path to write the JSON report."),
) -> None:
    """Load a real Transformers model, capture KV, encode/decode it, and report error."""

    try:
        report = run_transformers_kvcodec_artifact_benchmark(
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            artifact_dir=artifact_dir,
            layout_name=layout_name,
            layers_per_frame=layers_per_frame,
            codec_name=codec,
            quant_axis=quant_axis,
            nvenc_workers=nvenc_workers,
            nvdec_workers=nvdec_workers,
            device_map=device_map,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    payload = report.to_json()
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload + "\n", encoding="utf-8")
    typer.echo(payload)


@app.command("vllm-lmcache")
def vllm_lmcache_command(
    model: str = typer.Option(QWEN3_17B_MODEL, "--model", help="Model passed to vLLM serve."),
    mode: Literal["mp", "in-process"] = typer.Option("mp", "--mode", help="LMCache deployment mode: mp starts LMCache server; in-process loads LMCache inside vLLM."),
    prompt: str = typer.Option("KVCodec live vLLM LMCache benchmark prompt. Summarize KV cache reuse in one sentence.", "--prompt", help="Repeated prompt used to exercise LMCache hit path."),
    prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", exists=True, dir_okay=False, file_okay=True, help="UTF-8 file containing the repeated prompt; first non-empty line is used."),
    max_tokens: int = typer.Option(16, "--max-tokens", min=1, help="Max generated tokens per request."),
    requests: int = typer.Option(2, "--requests", min=1, help="Number of repeated streaming requests."),
    vllm_port: int = typer.Option(8000, "--vllm-port", min=1, help="vLLM HTTP port."),
    lmcache_port: int = typer.Option(5555, "--lmcache-port", min=1, help="LMCache ZMQ port."),
    lmcache_http_port: int = typer.Option(18080, "--lmcache-http-port", min=1, help="LMCache HTTP/health port."),
    lmcache_l1_gb: float = typer.Option(4.0, "--lmcache-l1-gb", min=0.1, help="LMCache L1 size in GB."),
    startup_timeout_s: float = typer.Option(900.0, "--startup-timeout-s", min=1.0, help="Max seconds to wait for vLLM model startup."),
    request_timeout_s: float = typer.Option(300.0, "--request-timeout-s", min=1.0, help="Max seconds for one streaming request."),
    log_dir: Path = typer.Option(Path("live-vllm-lmcache-logs"), "--log-dir", file_okay=False, dir_okay=True, help="Directory for LMCache/vLLM logs."),
    lmcache_shipped_connector: bool = typer.Option(True, "--lmcache-shipped-connector/--vllm-builtin-connector", help="Use LMCache's connector module path instead of vLLM's built-in connector."),
    extra_vllm_arg: list[str] = typer.Option(None, "--extra-vllm-arg", help="Extra argument appended to `vllm serve`; repeat for multiple values."),
    kvcodec_storage: bool = typer.Option(False, "--kvcodec-storage/--no-kvcodec-storage", help="Enable cradle_codec as an LMCache storage_plugins backend."),
    kvcodec_plugin_name: str = typer.Option("cradle_codec", "--kvcodec-plugin-name", help="LMCache storage_plugins name for the KV codec backend."),
    kvcodec_artifact_root: Path = typer.Option(Path(".cradle-codec"), "--kvcodec-artifact-root", help="Artifact root used by the Cradle Codec LMCache storage backend."),
    kvcodec_codec: Literal["reference", "ffmpeg", "pynv", "pynvvideocodec"] = typer.Option("pynvvideocodec", "--kvcodec-codec", help="KVCodec frame backend used by the LMCache storage plugin."),
    kvcodec_quant_axis: Literal["part", "frame", "channel"] = typer.Option("channel", "--kvcodec-quant-axis", help="uint8_minmax quantization axis used by the storage plugin."),
    kvcodec_layout: Optional[str] = typer.Option(None, "--kvcodec-layout", help="Optional KVCodec layout candidate name, e.g. h1x8_d32x4_32x32."),
    kvcodec_layers_per_frame: Optional[int] = typer.Option(None, "--kvcodec-layers-per-frame", min=1, max=3, help="Optional layer grouping override for the storage plugin."),
    kvcodec_head_rows: Optional[int] = typer.Option(None, "--kvcodec-head-rows", min=1, help="Explicit layout head rows; provide all four tiling dimensions together."),
    kvcodec_head_cols: Optional[int] = typer.Option(None, "--kvcodec-head-cols", min=1, help="Explicit layout head columns; provide all four tiling dimensions together."),
    kvcodec_dim_rows: Optional[int] = typer.Option(None, "--kvcodec-dim-rows", min=1, help="Explicit layout head-dim rows; provide all four tiling dimensions together."),
    kvcodec_dim_cols: Optional[int] = typer.Option(None, "--kvcodec-dim-cols", min=1, help="Explicit layout head-dim columns; provide all four tiling dimensions together."),
    kvcodec_io_threads: Optional[int] = typer.Option(None, "--kvcodec-io-threads", min=1, help="Optional storage plugin I/O worker count."),
    kvcodec_nvenc_workers: Optional[int] = typer.Option(None, "--kvcodec-nvenc-workers", min=1, help="PyNvVideoCodec NVENC worker count for the storage plugin."),
    kvcodec_nvdec_workers: Optional[int] = typer.Option(None, "--kvcodec-nvdec-workers", min=1, help="PyNvVideoCodec NVDEC worker count for the storage plugin."),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional path to write the JSON report."),
) -> None:
    """Launch vLLM with LMCache and measure live streaming TTFT/TPOT."""
    if prompt_file is not None:
        file_prompts = _load_prompts((), prompt_file)
        if len(file_prompts) != 1:
            raise typer.BadParameter("--prompt-file for vllm-lmcache must contain exactly one non-empty line")
        prompt = file_prompts[0]
    try:
        report = run_vllm_lmcache_live_benchmark(
            model=model,
            mode=mode,
            prompt=prompt,
            max_tokens=max_tokens,
            requests=requests,
            vllm_port=vllm_port,
            lmcache_port=lmcache_port,
            lmcache_http_port=lmcache_http_port,
            lmcache_l1_gb=lmcache_l1_gb,
            startup_timeout_s=startup_timeout_s,
            request_timeout_s=request_timeout_s,
            log_dir=log_dir,
            use_lmcache_shipped_connector=lmcache_shipped_connector,
            extra_vllm_args=tuple(extra_vllm_arg or ()),
            kvcodec_storage_plugin=lmcache_kvcodec_storage_plugin_config(
                enabled=kvcodec_storage,
                plugin_name=kvcodec_plugin_name,
                artifact_root=kvcodec_artifact_root,
                codec=kvcodec_codec,
                quantization_axis=kvcodec_quant_axis,
                layout_name=kvcodec_layout,
                layers_per_frame=kvcodec_layers_per_frame,
                head_rows=kvcodec_head_rows,
                head_cols=kvcodec_head_cols,
                dim_rows=kvcodec_dim_rows,
                dim_cols=kvcodec_dim_cols,
                io_threads=kvcodec_io_threads,
                nvenc_workers=kvcodec_nvenc_workers,
                nvdec_workers=kvcodec_nvdec_workers,
            ),
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    payload = report.to_json()
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload + "\n", encoding="utf-8")
    typer.echo(payload)


if __name__ == "__main__":
    app()
