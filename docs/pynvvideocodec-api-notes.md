# pynvvideocodec 2.1 notes

reference captured from the provided nvidia api page plus local `uv run` introspection.

## module

- import: `import PyNvVideoCodec as nvc`.
- local package observed: `PyNvVideoCodec 2.1.0`.
- useful module attrs: `__version__`, `__cuda_version__`, `__video_codec_sdk_version__`.

## nvenc encode

use `nvc.CreateEncoder(width, height, fmt, usecpuinputbuffer, **kwargs)` in the installed 2.1 package.

common kwargs used here:

- `gpu_id=0`
- `codec="hevc"`
- `preset="P1"`
- `tuning_info="lossless"`
- `rc="constqp"`
- `constqp=0`
- `fps=30`
- `gop=30`
- `bf=0`

for 3-channel kv frames, pass frame planes as contiguous `uint8` yuv444, shape `[3, H, W]`, with `fmt="YUV444"` and `usecpuinputbuffer=True` unless the caller supplies gpu memory. call `encoder.Encode(frame)` per frame, then `encoder.EndEncode()` once to flush delayed packets. query capability with `nvc.GetEncoderCaps(gpu_id, "hevc")`; require `support_yuv444_encode` and `support_lossless_encode` for this backend.

## nvdec decode

for elementary hevc payloads, use low-level demux/decode rather than `SimpleDecoder`/`ThreadedDecoder`, because those high-level apis require seekable container files.

current elementary-stream path:

- `demuxer = nvc.CreateDemuxer(feed_callback)` where the callback fills a bytearray and returns byte count.
- `decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), usedevicememory=False, maxwidth=W, maxheight=H, outputColorType=nvc.OutputColorType.NATIVE, latency=nvc.DisplayDecodeLatencyType.ZERO)`.
- for each packet from `demuxer`, call `decoder.Decode(packet)`.
- after the demuxer is exhausted, call `decoder.Flush()`.
- convert returned `DecodedFrame` objects to numpy for the current cpu restore path. future paged restore should keep `usedevicememory=True` and consume cuda array interface / dlpack / plane pointers.

## threaded/high-level decoders

- `nvc.ThreadedDecoder(enc_file_path, buffer_size, ...)` prefetches from a seekable container path on a background thread.
- `nvc.SimpleDecoder(enc_file_path, ...)` supports indexing/slicing over seekable container video.
- these are not appropriate for the current `.bin` elementary hevc artifacts unless artifacts are changed to a container format.

## worker pool decision

current implementation should pool python worker threads around the low-level elementary-stream nvdec path and around per-part nvenc encode calls. it must not silently fall back to ffmpeg in the `pynvvideocodec` performance path; unsupported nvdec must raise a clear error. `ffmpeg` remains an explicit backend for compatibility/testing only.
