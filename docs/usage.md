# usage

this guide covers the cradle codec workflows that are wired today: local artifacts, explicit codec backends, remote artifact fetch over http, optional gstreamer packet transport, dependency inspection, and a live vllm/lmcache cache-hit path. see [architecture](architecture.md) for the data model and runtime boundaries.

## installation

the distribution and cli are named `cradle-codec`. the import package and lmcache plugin are named `cradle_codec`, with source under `src/cradle_codec`.

the core dependency set supports manifests, tensor layouts, the reference codec, local storage, and http fetch. optional extras add pynvvideocodec, gstreamer/pygobject bindings, pytorch, transformers, vllm, and lmcache.

```bash
cd cradle-codec
uv sync
uv run cradle-codec doctor --compact
```

for the nvidia and serving paths:

```bash
uv sync --extra gpu --extra serving
uv run cradle-codec doctor --compact
```

`doctor` reports package imports and local executables. it catches common setup failures, but it does not prove that the installed gpu and driver support every hevc mode. use the focused checks in [verification](verification.md) to exercise a backend.

## gstreamer packet transport

the `gstreamer` extra exposes the gdp-over-tcp sender and receiver under `cradle_codec.gstreamer`. install a gstreamer 1.0 runtime, the plugins that provide `gdppay`, `gdpdepay`, `tcpserversink`, and `tcpclientsrc`, the gst 1.0 typelib, and the native prerequisites needed to install pygobject. then install and probe the optional path:

```bash
uv sync --extra gstreamer
uv run cradle-codec doctor --compact
uv run cradle-codec gstreamer loopback
```

the loopback command starts both endpoints, waits for the tcp client to attach, sends one utf-8 packet, and verifies byte equality. this module is a packet-boundary transport primitive. it does not yet connect artifact fetch to the vllm request lifecycle, decode hevc through gstreamer, expose cuda decode surfaces, or restore directly into paged kv memory.

## artifact model

artifact commands use canonical kv arrays shaped `[2, L, T, H, D]`. the first axis selects k or v, `L` is layer count, `T` is cached-token count, `H` is kv-head count, and `D` is head dimension. encoding maps tokens to frame time, groups up to three adjacent layers into channels, and tiles the head/head-dimension plane into frame height and width without reordering values within a head.

each artifact contains `manifest.json` plus payloads under `parts/`. the manifest records the layout, canonical shape, quantization metadata, codec parameters, checksums, and source identity required to restore the tensor deterministically.

## encode and decode one kv chunk

use `cradle-codec encode` with a `.npy` array in canonical shape. for a qwen/qwen3-1.7b-style chunk with 8 kv heads and head dimension 128, the following tiling maps the head/head-dimension plane to a logical `1 x 1024` frame area. a video backend may pad the physical frame to satisfy hardware limits; restoration ignores that padding.

```bash
uv run cradle-codec encode \
  --input /path/to/kv.npy \
  --output /tmp/cradle-codec-artifact \
  --source-key qwen3/example/chunk-0 \
  --model Qwen/Qwen3-1.7B \
  --layers-per-frame 3 \
  --head-rows 1 --head-cols 8 \
  --dim-rows 1 --dim-cols 128 \
  --quant-mode uint8_minmax \
  --quant-axis channel \
  --codec pynvvideocodec \
  --nvenc-workers 2 --nvdec-workers 2
```

decode the artifact back to a canonical kv array:

```bash
uv run cradle-codec decode \
  --artifact /tmp/cradle-codec-artifact \
  --output /tmp/cradle-codec-restored.npy
```

the command prints the restored shape and dtype. `uint8_minmax` quantization does not preserve exact floating-point values; use reconstruction metrics and model-level evaluation to decide whether a setting is acceptable. the `reference` backend preserves the quantized frame bytes, not the original floating-point values, and is useful when isolating layout and artifact behavior.

## codec backends

- `pynvvideocodec` is the nvidia path. it uses pynvvideocodec for hevc yuv444 encode/decode through nvenc and nvdec on compatible hosts. small logical frames may be padded for hardware decode. an unsupported nvdec path raises an error instead of silently falling back.
- `ffmpeg` is an explicit compatibility and debugging path. it is not a substitute for gpu-path measurements.
- `reference` stores checksummed raw `uint8` frame bytes so layout, manifest, restore, storage, fetch, and selection behavior can be checked without video dependencies. it still exercises `uint8_minmax` quantization.

## estimate reuse cost from an artifact

`bench` operates on an existing artifact without starting a model. given a bandwidth model and full-prefill time, it estimates raw kv transfer and codec-artifact reuse costs and reports reconstruction metrics when `--expected` is supplied.

```bash
uv run cradle-codec bench \
  --artifact /tmp/cradle-codec-artifact \
  --expected /path/to/kv.npy \
  --bandwidth-bytes-per-sec 2000000000 \
  --prefill-ms 5000
```

if an artifact has named variants, omit `--variant` to let the deterministic selector choose from manifest metadata using the supplied bandwidth. this is a planning estimate, not live adaptation to measured network conditions.

## live vllm and lmcache smoke path

the live wrapper starts vllm, configures lmcache, sends repeated openai-compatible completion requests, and writes ttft/tpot timing json. with `--kvcodec-storage`, lmcache loads `cradle_codec` as a storage plugin. the initial request can populate artifacts; a repeated request should report cached tokens when the integration is working.

```bash
uv run cradle-codec live vllm-lmcache \
  --mode in-process \
  --model Qwen/Qwen3-1.7B \
  --prompt "KV cache reuse context sentence. What is repeated?" \
  --max-tokens 2 \
  --requests 2 \
  --startup-timeout-s 900 \
  --request-timeout-s 240 \
  --lmcache-l1-gb 0.1 \
  --log-dir /tmp/cradle-codec-live-smoke \
  --kvcodec-storage \
  --kvcodec-artifact-root /tmp/cradle-codec-live-artifacts \
  --kvcodec-codec pynvvideocodec \
  --kvcodec-nvenc-workers 2 \
  --kvcodec-nvdec-workers 2 \
  --extra-vllm-arg=--max-model-len \
  --extra-vllm-arg=2048 \
  --extra-vllm-arg=--gpu-memory-utilization \
  --extra-vllm-arg=0.75 \
  --extra-vllm-arg=--enforce-eager \
  --output-json /tmp/cradle-codec-live-smoke/report.json
```

for a longer prompt, put exactly one non-empty line in a prompt file and pass `--prompt-file`. the prompt must be long enough for lmcache to store a useful chunk; a very short prompt may run correctly without exercising meaningful cache storage.

a useful smoke run has three observable facts: the vllm log reports that the `cradle_codec` backend was created, the artifact root contains `.kvcodec` directories with manifests and part payloads, and the repeated request reports lmcache hit tokens. timing depends on model, prompt, cache state, gpu, driver, vllm, and lmcache versions; compare runs only when those inputs are controlled.

## print integration configuration

to launch vllm and lmcache outside the wrapper, print the generated environment and argument bundle:

```bash
uv run cradle-codec integration vllm-lmcache \
  --mode in-process \
  --model Qwen/Qwen3-1.7B \
  --kvcodec-storage \
  --kvcodec-artifact-root /tmp/cradle-codec-live-artifacts \
  --kvcodec-codec pynvvideocodec \
  --kvcodec-quant-axis channel \
  --kvcodec-nvenc-workers 2 \
  --kvcodec-nvdec-workers 2 \
  --launch-bundle
```

multi-process mode can also print lmcache server arguments and vllm `--kv-transfer-config` json. in-process mode loads lmcache inside vllm and is simpler for local smoke checks because it does not require a separate lmcache server process.

## serve and fetch artifacts over http

the remote service exposes a `LocalArtifactStore` root through a minimal read-only protocol. an lmcache run using `--kvcodec-artifact-root` naturally creates the expected store layout.

```bash
uv run cradle-codec remote serve \
  --root /tmp/cradle-codec-live-artifacts \
  --host 127.0.0.1 \
  --port 8080
```

fetch one source key, validate and decode its payloads, and write a restored array:

```bash
uv run cradle-codec remote fetch \
  --base-url http://127.0.0.1:8080 \
  --source-key '<lmcache-source-key>' \
  --output /tmp/cradle-codec-fetched-kv.npy \
  --codec pynvvideocodec \
  --bandwidth-bytes-per-sec 2000000000
```

the source key is the stable identity recorded at storage time. lmcache-derived keys include model, tensor format, token span, and cache identity information. prefer local encode/decode for artifact work that does not need the http boundary.

the remote path is suitable for controlled local and 10 gbe measurements. it does not yet overlap transfer with per-frame decode or continuously respond to changing wan conditions. those milestones are described in the [roadmap](roadmap.md).
