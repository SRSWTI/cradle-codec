# roadmap

cradle codec is an apache-2.0 research systems project focused on reliable kv-cache movement across storage, networks, and heterogeneous compute. the current implementation is deliberately inspectable: canonical tensors become self-describing artifacts, backends are explicit, corruption is checked, transport and decode timings are separated, and unsupported gpu paths fail visibly.

this roadmap moves from that working artifact layer toward a low-contention data-center and wan cache-transfer runtime. it records current capability separately from future milestones so engineering decisions and performance statements stay tied to observable behavior. see [architecture](architecture.md), [usage](usage.md), and [verification](verification.md) for the present system.

## current capabilities

| area | current state | boundary |
| --- | --- | --- |
| tensor layout | deterministic packing from canonical `[2, L, T, H, D]` kv arrays into token-time frame batches, with adjacent-layer channels and head/dimension tiling. | candidate helpers exist, but comprehensive model-specific search and automatic multi-resolution generation are not yet standard workflows. |
| quantization | `uint8_minmax` with `part`, `frame`, and `channel` metadata axes. | floating-point restoration is lossy; model-level quality needs controlled evaluation for each configuration. |
| codec backends | checksummed `reference`, explicit `ffmpeg`, and pynvvideocodec hevc yuv444 through nvenc/nvdec on compatible nvidia hosts. | hardware and driver coverage is incomplete, and the pynv decode path currently returns frames through python and numpy. |
| artifact and storage | manifest schema, independent payload parts, checksums, local artifact storage, named variants, deterministic encode/decode, and stable source keys. | artifact lifecycle management across distributed cache nodes is not implemented. |
| remote transport | read-only http manifest and payload fetch with separate transfer/decode/restore timings. | transfer is not yet overlapped with per-frame decode, and the service is not a production disaggregated-cache control plane. |
| variant selection | deterministic selection from manifest metadata, a supplied bandwidth estimate, and optional switch penalty. | selection does not yet use live transport and decode measurements. |
| lmcache/vllm | python lmcache storage plugin, launch/config helpers, live repeated-request smoke path, and artifact restoration into lmcache-compatible objects. | integration is storage-oriented rather than a complete remote-fetch request lifecycle. |
| scheduling | a deterministic fetch-scheduling model supports focused logic checks. | fetch state and wakeups are not wired into the live vllm scheduler. |
| network environment | local and http workflows provide a practical baseline for controlled 10 gbe evaluation. | wan behavior with shaped bandwidth, latency, jitter, and loss has not yet been established. |

## milestone 1: native c++/gstreamer nvdec pipeline

build a c++ pipeline around gstreamer and the nvidia decode stack so encoded parts can move from transport buffers through nvdec without python owning the hot path. the pipeline must:

- maintain a bounded pool aligned with available nvdec capacity;
- accept independently decodable artifact parts and surface format or hardware failures without cpu fallback;
- emit decoded frames incrementally instead of materializing a complete decoded chunk;
- define ownership and backpressure between network receive, demux, decode, and restoration stages;
- expose queue depth, throughput, engine utilization, decode errors, and cancellation behavior;
- cover representative data-center and workstation nvidia gpus, driver versions, and supported hevc yuv444 modes.

this milestone is about compute reliability as much as throughput. a production path must remain diagnosable under malformed payloads, decoder resets, resource exhaustion, request cancellation, and mixed hardware fleets.

## milestone 2: frame-wise restoration and direct vllm paged-memory writes

replace full-batch python/numpy restoration on the serving hot path with frame-wise dequantization and repacking into preallocated vllm kv-cache slots. each decoded frame should be written directly to its destination pages as soon as its dependencies are satisfied.

required work includes:

- a stable mapping from artifact side/layer/token coordinates to vllm block-table destinations;
- a cuda kernel or compatible vllm extension for sparse writes into paged kv memory;
- dtype conversion and dequantization without a full restored-chunk allocation;
- ordering, completion, cancellation, and partial-failure semantics across frame writes;
- destination lifetime management so pages cannot be reused while decode is in flight;
- correctness checks for partial chunks, non-contiguous page assignments, model layouts, and boundary token counts.

the milestone is complete only when the live serving path can restore decoded frames into paged memory without routing the whole chunk through a numpy kv array.

## milestone 3: full lmcache/vllm request-lifecycle integration

move beyond the storage plugin and integrate remote-cache reuse through the complete serving request lifecycle. the live system needs:

- remote-cache discovery and artifact-variant selection before prefill decisions are finalized;
- explicit fetch-request states for queued, transferring, decoding, restoring, ready, cancelled, and failed work;
- scheduler isolation so network-bound reuse does not stall ordinary requests;
- preallocated destination slots and wakeups tied to verified restoration completion;
- cancellation and fallback policy that preserves request correctness when remote state is late, corrupt, or unavailable;
- lmcache metadata and eviction coordination so artifact identity, token coverage, and model configuration cannot drift;
- end-to-end telemetry linking cache lookup, transfer, decode, restore, scheduler delay, and ttft.

the integration must preserve serving correctness first. recomputing a prefix after an explicit fetch failure is acceptable policy; silently exposing partial or mismatched kv state is not.

## milestone 4: adaptive transport and bandwidth-shaped wan validation

use the current 10 gbe environment as a controlled data-center baseline, then validate behavior beyond it with shaped wan profiles. the harness should vary bandwidth, round-trip latency, jitter, loss, reordering where relevant, and competing traffic while replaying controlled request traces.

this milestone includes:

- automatic creation of multiple layout or resolution variants for the same source chunk;
- online selection from measured transfer, decode, and restore state rather than a supplied scalar alone;
- overlap of payload transfer, incremental decode, and paged-memory restoration;
- switching and hysteresis policies that avoid oscillation as link conditions change;
- experiments below, at, and above 10 gbe plus high-latency wan cases;
- failure and recovery checks for timeouts, truncated payloads, checksum errors, decoder faults, and cache eviction;
- reports that separate payload size, effective throughput, queueing, decode, restoration, scheduler delay, cache-hit coverage, and end-to-end ttft.

the goal is not a single headline number. it is a defensible operating envelope showing when compression and remote reuse help, when local recomputation is preferable, and how the decision changes across networks and hardware.

## milestone 5: layout, quality, and heterogeneous-hardware evaluation

make layout and quantization choices repeatable across models rather than relying on one configuration. capture real kv chunks, evaluate candidate tilings under fixed codec and quantization settings, and report artifact size, encode/decode cost, reconstruction error, and downstream generation quality.

run the same contracts across heterogeneous hardware: data-center accelerators, supported workstation gpus, multiple driver families, and cpu compatibility environments. capability reporting should identify package import, nvenc, nvdec, demux, frame conversion, format, and minimum-dimension failures separately.

evaluation should control model, context length, chunk size, dtype, quantization axis, layout, codec settings, software versions, hardware, cache state, and network profile. results must distinguish artifact-level compression from serving-level benefit.

## engineering guideposts

- keep backends explicit. the nvidia path must never become a cpu path silently.
- validate checksums and model/layout identity before restored state reaches a request.
- prefer bounded queues, explicit ownership, and cancellation over unbounded background work.
- measure conditional behavior and failure recovery, not configuration defaults.
- preserve the artifact boundary unless an end-to-end measurement demonstrates that a change requires a new contract.
- treat 10 gbe results as a data-center baseline, not evidence for wan behavior.
- tie serving claims to complete request traces and quality checks, not codec ratios in isolation.

near-term contributions can improve capability diagnostics, add bounded layout candidates, extend manifest validation, or document one hardware configuration. the load-bearing systems work is the c++/gstreamer nvdec pipeline, direct frame-wise writes into vllm paged memory, full lmcache/vllm lifecycle integration, and bandwidth-shaped wan validation.
