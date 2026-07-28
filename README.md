# cradle codec

cradle codec is gpu-native kv cache compression and streaming infrastructure for long-context inference.

we started this work because kv cache storage became a real bottleneck in our own production pipelines. reusable prefixes save compute, but the resulting state is large, expensive to retain, and slow to move between machines. once cache reuse crosses a host boundary, storage and networking become part of the inference path.

we are building cradle codec as a serious systems research effort around that problem.

## the problem

long-context inference produces a large key-value cache for every reusable prefix. keeping all of it in accelerator memory is rarely practical. moving raw tensors to local storage, another node, or a remote cache service can erase the compute saved by reuse.

our current environment has 10 gbe connectivity. that is fast enough to make remote cache reuse useful, but slow enough that raw kv transfer remains expensive. compression changes the operating point: less data crosses the link, and dedicated video hardware can perform the codec work without directly consuming the same cuda cores as model execution.

this is the first step toward portable inference state across machines, accelerators, data centers, and eventually the wan.

## our research thesis

we treat kv cache as structured temporal data, not an opaque byte array.

neighboring tokens produce related state. model layers and attention heads give that state a stable shape. by preserving those relationships when mapping tensors into frames, a video codec can remove redundancy across both space and time.

our design follows four principles:

- preserve token order as video time
- keep attention heads structurally separated inside each frame
- map adjacent layers into independent frame channels
- use dedicated codec engines for the encoded transport format

this creates a clean separation between model execution and cache transport. encoding can happen offline or outside the critical request path. compressed artifacts can live on disk or behind a storage service. decode and restoration can be moved progressively closer to the serving engine.

## how it works

```text
canonical kv tensor
    -> head-preserving frame layout
    -> uint8 quantization and metadata
    -> lossless hevc encode
    -> checksummed cache artifact
    -> local or remote transfer
    -> hevc decode
    -> dequantization and layout restore
    -> lmcache or serving-engine memory
```

the canonical tensor shape is:

```text
[2, layers, tokens, kv heads, head dimension]
```

key and value tensors remain separate. token position becomes frame position in time. up to three adjacent layers occupy the three frame channels. the remaining head and head-dimension axes are tiled into the frame plane without mixing head identity.

## what works today

- deterministic tensor-to-frame packing and inverse restoration
- explicit head and head-dimension tiling
- uint8 min/max quantization with stored restoration metadata
- lossless hevc through pynvvideocodec on supported nvidia hardware
- an ffmpeg compatibility backend
- a dependency-free reference backend for core verification
- checksummed, self-describing cache artifacts
- local artifact storage and read-only http transport
- partial and full artifact restoration
- an optional gstreamer gdp-over-tcp byte-packet transport and loopback command
- multiple artifact variants with cost-based selection
- an lmcache storage plugin
- vllm and lmcache launch configuration
- a live repeated-request cache-hit path
- local reports for encoded bytes, decode time, restore time, and reconstruction error

## why video hardware

modern accelerators often include media engines beside their general-purpose compute cores. those engines are built for high-throughput prediction, entropy coding, and frame reconstruction. during language-model inference they may otherwise remain underused.

cradle codec turns that hardware asymmetry into a systems primitive. the long-term goal is not video for its own sake. the goal is a compact, hardware-accelerated representation of reusable inference state that can move through real storage and networking systems.

## quick start

install the core package:

```bash
uv sync --group dev
uv run cradle-codec doctor --compact
uv run pytest -q
```

encode and restore a canonical kv chunk:

```bash
uv run cradle-codec encode \
  --input /path/to/kv.npy \
  --output /tmp/cradle-codec-artifact \
  --source-key example/chunk-0 \
  --model example/model \
  --layers-per-frame 3 \
  --head-rows 1 --head-cols 8 \
  --dim-rows 1 --dim-cols 128 \
  --codec pynvvideocodec

uv run cradle-codec decode \
  --artifact /tmp/cradle-codec-artifact \
  --output /tmp/cradle-codec-restored.npy
```

install the gpu and serving paths:

```bash
uv sync --extra gpu --extra serving --group dev
uv run cradle-codec live vllm-lmcache --help
```

install the optional gstreamer transport after installing the gstreamer 1.0 runtime, plugins-base elements, typelib, and pygobject build prerequisites:

```bash
uv sync --extra gstreamer --group dev
uv run cradle-codec gstreamer loopback
```

to build the bundled gstreamer development tree, first install a c/c++ toolchain, `pkg-config`, and the native prerequisites for pygobject. cradle codec's python gstreamer integration requires pygobject as well, so install the `gstreamer` extra before building:

```bash
uv sync --extra gstreamer --group dev
cd src/cradle_codec/helpers/gstreamer
./setup_and_build.sh
uv run meson devenv -C builddir
```

the setup script creates the documentation metadata required by the source tree, configures a clean `builddir` with documentation disabled, and compiles gstreamer. the final command opens a development shell with the newly built libraries, plugins, and tools on the correct paths.

all bundled dependency definitions, upstream build documentation, and the material needed to configure gstreamer and `appsink` are kept under `src/cradle_codec/helpers/gstreamer`. start with the [bundled gstreamer readme](src/cradle_codec/helpers/gstreamer/README.md), then use the [gstapp documentation](src/cradle_codec/helpers/gstreamer/subprojects/gst-plugins-base/docs/libs/app/index.md) and [appsrc/appsink examples](src/cradle_codec/helpers/gstreamer/subprojects/gst-plugins-base/tests/examples/app/) for application-owned pipelines.

inspect the full workflows in [usage](docs/usage.md) and [verification](docs/verification.md).

## current boundary

this repository has a real artifact path and a real lmcache storage integration. it does not yet place each decoded frame directly into vllm paged kv memory. the current serving path restores completed batches through python, numpy, and lmcache memory objects.

the current gstreamer module carries complete byte packets through `appsrc`, gdp framing, tcp, and `appsink`. it is a usable transport primitive, not the future c++ nvdec hot path: it does not decode hevc frames, expose cuda surfaces, or write vllm paged memory.

quantization is also intentionally approximate. lossless hevc means the codec preserves the quantized integer frames; it does not mean the original floating-point tensor is bit-exact after quantization. reconstruction error and model quality must both be measured.

these boundaries are explicit because reliability matters more than a benchmark headline.

## where this is going

### serving hot path

- a c++ and gstreamer nvdec pipeline
- frame callbacks as decoded surfaces become available
- frame-wise dequantization and layout restoration
- direct writes into vllm paged kv memory
- no full decoded chunk in host memory
- bounded decoder pools with explicit backpressure

### integration

- complete lmcache and vllm request-lifecycle integration
- remote cache discovery and destination-slot allocation
- isolated scheduling for cache-hit and cache-miss traffic
- failure handling that can recompute safely when fetch fails
- observable cache provenance, checksums, and restoration state

### networking

- controlled 10 gbe transfer experiments
- shaped bandwidth and jitter tests
- overlapping transfer, decode, and restoration
- adaptive artifact resolution and decode-pool selection
- cross-region and wan kv cache movement
- heterogeneous producer and consumer hardware

### evaluation

- real long-context workloads
- model-level quality checks after cache restoration
- compression and latency comparisons under matched conditions
- memory-pressure and concurrent-request experiments
- reliability testing for partial, corrupt, missing, and stale artifacts

## research direction

we are a research outfit working on compute reliability, networking, data-center systems, and heterogeneous hardware. cradle codec is one part of a larger effort to make inference state durable, inspectable, and movable instead of tying it permanently to one accelerator process.

we believe the next generation of inference systems will treat reusable state as a distributed systems object. it will have storage policy, transport formats, integrity metadata, placement decisions, and hardware-aware execution paths. cradle codec is our path toward that system.

see [architecture](docs/architecture.md), [usage](docs/usage.md), [verification](docs/verification.md), and the [roadmap](docs/roadmap.md).

## inspiration and distinction

we thank ceyu xu, yongji wu, xinyu yang, boyuan chen, matthew lentz, danyang zhuo, and lisa w. wills, whose work, [llm.265: video codecs are secretly tensor codecs](https://doi.org/10.1145/3725843.3756078), was one of our inspirations. it strengthened the broader case for applying video codecs to tensors, but it did not provide the systems path we needed for real inference stacks: artifact storage, network transport, cache lookup, request-lifecycle integration, failure handling, and restoration into serving memory.

our core direction was developed independently: nccl-like infrastructure for streaming live inference state over ethernet, with codec compression and storage integrated into online serving. cradle codec is not only a tensor-compression experiment; it is an attempt to move reusable kv state through real networks and production inference runtimes.

we also thank the [gstreamer team](https://gstreamer.freedesktop.org/) for their exceptional work on a composable, production-grade media framework. cradle codec builds on that engineering through `appsrc`, `appsink`, gdp framing, tcp transport elements, development environments, and hardware-codec integration. their work turns difficult media and transport plumbing into dependable systems building blocks.

## license

apache-2.0.
