# verification guide

this guide verifies the capabilities implemented in the repository today: artifact packing, quantization, codec encode/decode, manifest validation, remote artifact fetch, optional gstreamer packet transport, lmcache storage integration, and a live vllm cache-hit smoke path. start with the smallest relevant tier and move toward the live serving path only after its prerequisites pass. for operating details, see [usage](usage.md); for subsystem boundaries, see [architecture](architecture.md).

## environment tiers

the core tier needs the default dependencies and covers deterministic artifact and reference-codec checks. the video tier adds either an `ffmpeg` executable or pynvvideocodec. the gstreamer tier adds the gstreamer 1.0 runtime, gdp/tcp elements, gst typelib, and pygobject. the serving tier adds pytorch, transformers, vllm, and lmcache. keeping the tiers separate makes package, driver, codec, transport, model-loading, and serving failures easier to distinguish.

```bash
cd cradle-codec
uv sync
uv run cradle-codec doctor --compact
```

for nvidia and serving checks:

```bash
uv sync --extra gpu --extra serving
uv run cradle-codec doctor --compact
nvidia-smi
```

for the gstreamer packet-transport check:

```bash
uv sync --extra gstreamer
uv run cradle-codec doctor --compact
uv run cradle-codec gstreamer loopback
```

the doctor report requires both the python bindings and `gst-launch-1.0`; the runtime probe additionally checks `appsrc`, `gdppay`, `tcpserversink`, `tcpclientsrc`, `gdpdepay`, and `appsink`. a successful loopback proves one complete packet crossed the local gdp/tcp pipeline byte-for-byte. it does not exercise hevc, nvdec, cuda memory, remote artifact selection, or vllm.


pynvvideocodec compatibility depends on the installed package build, driver, cuda runtime, gpu generation, and hevc format support. a successful `doctor` result confirms that the package is importable; it does not establish that every nvenc/nvdec yuv444 path works on the host. the focused pynv check below exercises that path.

## unit suite

run the unit suite from the project root:

```bash
uv run python -m unittest discover -s tests
```

a successful run ends with `OK`. gpu- and serving-specific checks may skip when optional packages or compatible hardware are unavailable; they must not turn an unsupported hardware path into an implicit fallback. run those tiers explicitly on the hosts where they matter.

## local artifact round trip

this sequence creates a synthetic canonical kv chunk, writes an artifact with the reference backend, and restores the array. it isolates layout, manifest, checksum, storage, and restoration behavior from video dependencies while retaining the normal `uint8_minmax` quantization path.

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np

rng = np.random.default_rng(7)
kv = rng.normal(size=(2, 4, 16, 2, 8)).astype(np.float32)
Path('/tmp/cradle-codec-verification').mkdir(parents=True, exist_ok=True)
np.save('/tmp/cradle-codec-verification/kv.npy', kv)
PY

uv run cradle-codec encode \
  --input /tmp/cradle-codec-verification/kv.npy \
  --output /tmp/cradle-codec-verification/artifact \
  --source-key verification/synthetic-chunk \
  --model synthetic \
  --layers-per-frame 2 \
  --head-rows 1 --head-cols 2 \
  --dim-rows 1 --dim-cols 8 \
  --codec reference

uv run cradle-codec decode \
  --artifact /tmp/cradle-codec-verification/artifact \
  --output /tmp/cradle-codec-verification/restored.npy

uv run python - <<'PY'
import numpy as np
original = np.load('/tmp/cradle-codec-verification/kv.npy')
restored = np.load('/tmp/cradle-codec-verification/restored.npy')
print('shape', restored.shape)
print('max_abs_error', np.max(np.abs(original - restored)))
PY
```

expect the restored shape to match the original. the `reference` backend preserves the quantized `uint8` frames exactly, but `uint8_minmax` remains lossy relative to the original float tensor, so `max_abs_error` is expected to be nonzero. use the same error interpretation for hevc backends; the additional codec stage is configured to preserve the quantized pixels.

## pynvvideocodec hevc check

run the pynv-specific test directly on a host where you need to establish nvidia codec compatibility:

```bash
uv run python tests/test_pynv_hevc.py
```

on a compatible host, the focused check must encode and decode a padded hevc yuv444 payload through pynvvideocodec. if nvdec reports a hardware error, keep that failure visible. the `pynvvideocodec` backend must not hide unsupported nvdec by falling back to ffmpeg.

## real-model artifact check

`live transformers-kv` loads a hugging face model through transformers, captures its kv cache, encodes and decodes an artifact, and reports reconstruction metrics. it exercises real kv distributions without starting vllm and lmcache.

```bash
uv run cradle-codec live transformers-kv \
  --model Qwen/Qwen3-1.7B \
  --prompt "Cradle Codec checks a real model KV cache." \
  --max-new-tokens 2 \
  --artifact /tmp/cradle-codec-transformers-artifact \
  --codec pynvvideocodec \
  --layers-per-frame 3 \
  --quant-axis channel \
  --nvenc-workers 2 \
  --nvdec-workers 2 \
  --output-json /tmp/cradle-codec-transformers-report.json
```

use `--codec reference` to isolate model loading and layout restoration from video behavior. use `--codec ffmpeg` only when intentionally checking the compatibility backend.

## live vllm and lmcache smoke

the prompt must be large enough for lmcache to store at least one useful chunk. this sequence writes one prompt line and sends it twice. the initial request pays the prefill and storage costs; the repeated request should report lmcache hit tokens.

```bash
uv run python - <<'PY'
from pathlib import Path
text = 'KV cache reuse context sentence. ' * 220 + 'What is repeated?\n'
Path('/tmp/cradle-codec-smoke-prompt.txt').write_text(text, encoding='utf-8')
PY

uv run cradle-codec live vllm-lmcache \
  --mode in-process \
  --model Qwen/Qwen3-1.7B \
  --prompt-file /tmp/cradle-codec-smoke-prompt.txt \
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

a successful live run produces `.kvcodec` artifacts and reports lmcache hit tokens on the repeated request. timing and restored shapes depend on model configuration, prompt length, chunk size, cache state, gpu, and driver behavior.

inspect the report and artifacts:

```bash
cat /tmp/cradle-codec-live-smoke/report.json
find /tmp/cradle-codec-live-artifacts -name manifest.json -print
```

logs should include the equivalent of `Created backend: cradle_codec`, successful token storage, and lmcache hit tokens. if those facts are absent, resolve the storage-plugin path before interpreting latency.

## decode an artifact created during serving

choose one `.kvcodec` directory from the live artifact root and decode it with the standalone cli:

```bash
uv run cradle-codec decode \
  --artifact /tmp/cradle-codec-live-artifacts/<artifact-name>.kvcodec \
  --output /tmp/cradle-codec-live-decoded.npy
```

a qwen/qwen3-1.7b lmcache chunk with 256 tokens commonly restores as `(2, 28, 256, 8, 128)`. other models and chunk sizes restore to their own canonical shape.

## capability boundary

the checks above cover current artifact behavior, explicit codec compatibility, http transport, the optional gstreamer gdp/tcp packet primitive, and the storage-oriented lmcache/vllm path. they do not establish direct frame-wise writes into vllm paged memory, a production c++/gstreamer nvdec pipeline, complete request-lifecycle fetch scheduling, or operation under bandwidth-shaped wan conditions. those remain explicit [roadmap](roadmap.md) milestones.
