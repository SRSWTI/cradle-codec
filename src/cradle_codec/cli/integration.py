from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer

from cradle_codec.integration import (
    QWEN3_17B_MODEL,
    instantiate_vllm_kv_transfer_config,
    lmcache_inprocess_kv_transfer_config,
    lmcache_kvcodec_storage_env,
    lmcache_kvcodec_storage_lmcache_config,
    lmcache_kvcodec_storage_plugin_config,
    lmcache_mp_kv_transfer_config,
    lmcache_server_args,
    vllm_serve_args,
)


app = typer.Typer(help="Print dependency-light vLLM/LMCache integration configs.", no_args_is_help=True)


def _lmcache_server_host(connector_host: str) -> str:
    return connector_host.removeprefix("tcp://")


@app.command("vllm-lmcache")
def vllm_lmcache_command(
    model: str = typer.Option(QWEN3_17B_MODEL, "--model", help="Model name passed to `vllm serve`."),
    mode: Literal["mp", "in-process"] = typer.Option("mp", "--mode", help="LMCache deployment mode."),
    port: int = typer.Option(8000, "--port", min=1, help="vLLM HTTP port."),
    role: Literal["kv_producer", "kv_consumer", "kv_both"] = typer.Option("kv_both", "--role", help="vLLM KV transfer role."),
    lmcache_host: str = typer.Option("tcp://localhost", "--lmcache-host", help="LMCache MP ZMQ host, including transport prefix."),
    lmcache_port: int = typer.Option(5555, "--lmcache-port", min=1, help="LMCache MP ZMQ port."),
    lmcache_server_url: list[str] | None = typer.Option(None, "--lmcache-server-url", help="LMCache MP server URL; repeat to use vLLM's multi-server connector path."),
    lmcache_mq_timeout_s: float | None = typer.Option(None, "--lmcache-mq-timeout-s", min=0.000001, help="LMCache MP request queue timeout in seconds."),
    lmcache_heartbeat_interval_s: float | None = typer.Option(None, "--lmcache-heartbeat-interval-s", min=0.000001, help="LMCache MP heartbeat interval in seconds."),
    lmcache_mp_transfer_mode: Literal["auto", "engine_driven", "lmcache_driven"] | None = typer.Option(None, "--lmcache-mp-transfer-mode", help="LMCache MP transfer scheduling mode."),
    lmcache_http_port: int = typer.Option(18080, "--lmcache-http-port", min=1, help="LMCache MP HTTP/health port included in launch bundles."),
    lmcache_l1_gb: float = typer.Option(4.0, "--lmcache-l1-gb", min=0.000001, help="LMCache L1 size in GB included in launch bundles."),
    lmcache_shipped_connector: bool = typer.Option(False, "--lmcache-shipped-connector", help="Use LMCache's external MP connector module path for vLLM versions that support it."),
    kvcodec_storage: bool = typer.Option(False, "--kvcodec-storage/--no-kvcodec-storage", help="Enable cradle_codec as an LMCache storage_plugins backend."),
    kvcodec_plugin_name: str = typer.Option("cradle_codec", "--kvcodec-plugin-name", help="LMCache storage_plugins name for the KV codec backend."),
    kvcodec_artifact_root: Path = typer.Option(Path(".cradle-codec"), "--kvcodec-artifact-root", help="Artifact root used by the Cradle Codec LMCache storage backend."),
    kvcodec_codec: Literal["reference", "ffmpeg", "pynv", "pynvvideocodec"] = typer.Option("pynvvideocodec", "--kvcodec-codec", help="KVCodec frame backend used by the LMCache storage plugin."),
    kvcodec_quant_axis: Literal["part", "frame", "channel"] = typer.Option("channel", "--kvcodec-quant-axis", help="uint8_minmax quantization axis used by the storage plugin."),
    kvcodec_layout: str | None = typer.Option(None, "--kvcodec-layout", help="Optional KVCodec layout candidate name, e.g. h1x8_d32x4_32x32."),
    kvcodec_layers_per_frame: int | None = typer.Option(None, "--kvcodec-layers-per-frame", min=1, max=3, help="Optional layer grouping override for the storage plugin."),
    kvcodec_head_rows: int | None = typer.Option(None, "--kvcodec-head-rows", min=1, help="Explicit layout head rows; provide all four tiling dimensions together."),
    kvcodec_head_cols: int | None = typer.Option(None, "--kvcodec-head-cols", min=1, help="Explicit layout head columns; provide all four tiling dimensions together."),
    kvcodec_dim_rows: int | None = typer.Option(None, "--kvcodec-dim-rows", min=1, help="Explicit layout head-dim rows; provide all four tiling dimensions together."),
    kvcodec_dim_cols: int | None = typer.Option(None, "--kvcodec-dim-cols", min=1, help="Explicit layout head-dim columns; provide all four tiling dimensions together."),
    kvcodec_io_threads: int | None = typer.Option(None, "--kvcodec-io-threads", min=1, help="Optional storage plugin I/O worker count."),
    kvcodec_nvenc_workers: int | None = typer.Option(None, "--kvcodec-nvenc-workers", min=1, help="PyNvVideoCodec NVENC worker count for the storage plugin."),
    kvcodec_nvdec_workers: int | None = typer.Option(None, "--kvcodec-nvdec-workers", min=1, help="PyNvVideoCodec NVDEC worker count for the storage plugin."),
    launch_bundle: bool = typer.Option(False, "--launch-bundle", help="Print env, LMCache server argv, and vLLM serve argv as JSON."),
    lmcache_config: bool = typer.Option(False, "--lmcache-config", help="Print LMCache storage plugin config/env instead of KV transfer config."),
    validate_runtime: bool = typer.Option(False, "--validate-runtime", help="Instantiate vLLM's real KVTransferConfig before printing; requires vLLM installed."),
    serve_args: bool = typer.Option(False, "--serve-args", help="Print a JSON argv array for `vllm serve` instead of only the KV transfer config."),
) -> None:
    """Print real vLLM KV transfer config JSON for LMCache-backed serving."""

    storage_plugin = lmcache_kvcodec_storage_plugin_config(
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
    )

    if mode == "mp":
        config = lmcache_mp_kv_transfer_config(
            host=lmcache_host,
            port=lmcache_port,
            server_urls=lmcache_server_url,
            role=role,
            use_lmcache_shipped_connector=lmcache_shipped_connector,
            mq_timeout_s=lmcache_mq_timeout_s,
            heartbeat_interval_s=lmcache_heartbeat_interval_s,
            mp_transfer_mode=lmcache_mp_transfer_mode,
        )
    else:
        config = lmcache_inprocess_kv_transfer_config(role=role)

    if validate_runtime:
        try:
            instantiate_vllm_kv_transfer_config(config)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if launch_bundle:
        typer.echo(
            json.dumps(
                {
                    "env": lmcache_kvcodec_storage_env(storage_plugin) if storage_plugin.enabled else {},
                    "lmcache_config": lmcache_kvcodec_storage_lmcache_config(storage_plugin),
                    "lmcache_server_args": None
                    if mode == "in-process"
                    else list(
                        lmcache_server_args(
                            host=_lmcache_server_host(lmcache_host),
                            port=lmcache_port,
                            http_port=lmcache_http_port,
                            l1_size_gb=lmcache_l1_gb,
                        )
                    ),
                    "vllm_serve_args": list(vllm_serve_args(model=model, port=port, kv_transfer_config=config)),
                },
                sort_keys=True,
                indent=2,
            )
        )
    elif lmcache_config:
        typer.echo(
            json.dumps(
                {
                    "env": lmcache_kvcodec_storage_env(storage_plugin) if storage_plugin.enabled else {},
                    "lmcache_config": lmcache_kvcodec_storage_lmcache_config(storage_plugin),
                },
                sort_keys=True,
                indent=2,
            )
        )
    elif serve_args:
        typer.echo(json.dumps(vllm_serve_args(model=model, port=port, kv_transfer_config=config), indent=2))
    else:
        typer.echo(json.dumps(config.to_dict(), sort_keys=True, indent=2))
