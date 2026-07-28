from __future__ import annotations

import threading
from dataclasses import dataclass
from time import monotonic, perf_counter, sleep
from typing import Any, Protocol

from .runtime import GStreamerRuntime, require_gstreamer


class PacketConsumer(Protocol):
    def consume_packet_bytes(self, packet_bytes: bytes) -> object:
        """Consume one complete packet emitted by the sender."""


@dataclass(frozen=True, slots=True)
class GStreamerSendResult:
    accepted: bool
    flow_return: str
    elapsed_ms: float


def _validated_port(port: int) -> int:
    port = int(port)
    if not 1 <= port <= 65_535:
        raise ValueError(f"port must be between 1 and 65535, got {port}")
    return port


def _make_element(runtime: GStreamerRuntime, factory: str, name: str) -> Any:
    element = runtime.Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"GStreamer could not construct element {factory!r}")
    return element


def _add_and_link(pipeline: Any, elements: tuple[Any, ...]) -> None:
    for element in elements:
        pipeline.add(element)
    for source, destination in zip(elements, elements[1:]):
        if not source.link(destination):
            raise RuntimeError(
                f"GStreamer could not link {source.get_name()!r} to {destination.get_name()!r}"
            )


class GStreamerPacketSender:
    """Send byte packets over a GStreamer GDP/TCP server pipeline."""

    def __init__(self, *, bind_host: str = "0.0.0.0", port: int = 42_425) -> None:
        if not bind_host:
            raise ValueError("bind_host must not be empty")
        self._port = _validated_port(port)
        self._runtime = require_gstreamer()
        Gst = self._runtime.Gst
        self._closed = False

        pipeline = Gst.Pipeline.new("cradle-codec-gstreamer-sender")
        if pipeline is None:
            raise RuntimeError("GStreamer could not construct the sender pipeline")
        self._pipeline = pipeline
        self._appsrc = _make_element(self._runtime, "appsrc", "packet-source")
        payloader = _make_element(self._runtime, "gdppay", "packet-payloader")
        self._tcp_sink = _make_element(self._runtime, "tcpserversink", "tcp-server")
        self._tcp_sink.set_property("host", bind_host)
        self._tcp_sink.set_property("port", self._port)
        self._tcp_sink.set_property("sync", False)

        caps = Gst.Caps.from_string("application/x-cradle-codec-packet")
        if caps is None:
            raise RuntimeError("GStreamer could not construct packet caps")
        self._appsrc.set_property("caps", caps)
        self._appsrc.set_property("format", Gst.Format.BYTES)
        self._appsrc.set_property("block", True)
        _add_and_link(self._pipeline, (self._appsrc, payloader, self._tcp_sink))

        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                "GStreamer sender pipeline failed to enter PLAYING state"
            )

    @property
    def port(self) -> int:
        return self._port

    def wait_for_clients(self, count: int = 1, *, timeout_s: float = 5.0) -> bool:
        if count <= 0:
            raise ValueError("count must be positive")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if self._closed:
            raise RuntimeError("GStreamer sender is closed")

        deadline = monotonic() + timeout_s
        while int(self._tcp_sink.get_property("num-handles")) < count:
            if monotonic() >= deadline:
                return False
            sleep(min(0.01, deadline - monotonic()))
        return True

    def send(self, packet: bytes) -> GStreamerSendResult:
        if self._closed:
            raise RuntimeError("GStreamer sender is closed")
        if not isinstance(packet, bytes):
            raise TypeError(f"packet must be bytes, got {type(packet).__name__}")

        started = perf_counter()
        buffer = self._runtime.Gst.Buffer.new_wrapped(packet)
        flow_return = self._appsrc.emit("push-buffer", buffer)
        elapsed_ms = (perf_counter() - started) * 1_000.0
        accepted = flow_return == self._runtime.Gst.FlowReturn.OK
        flow_name = getattr(flow_return, "value_nick", str(flow_return))
        return GStreamerSendResult(
            accepted=accepted,
            flow_return=str(flow_name),
            elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._appsrc.emit("end-of-stream")
        self._pipeline.set_state(self._runtime.Gst.State.NULL)

    def __enter__(self) -> GStreamerPacketSender:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


class GStreamerPacketReceiver:
    """Receive complete byte packets from a GStreamer GDP/TCP pipeline."""

    def __init__(self, host: str, port: int, consumer: PacketConsumer) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not callable(getattr(consumer, "consume_packet_bytes", None)):
            raise TypeError("consumer must define consume_packet_bytes(packet_bytes)")

        self._host = host
        self._port = _validated_port(port)
        self._consumer = consumer
        self._runtime = require_gstreamer()
        Gst = self._runtime.Gst
        self._loop = self._runtime.GLib.MainLoop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"cradle-gstreamer-receiver-{self._port}",
            daemon=True,
        )
        self._started = threading.Event()
        self._condition = threading.Condition()
        self._received_count = 0
        self._failure: RuntimeError | None = None
        self._closed = False

        pipeline = Gst.Pipeline.new("cradle-codec-gstreamer-receiver")
        if pipeline is None:
            raise RuntimeError("GStreamer could not construct the receiver pipeline")
        self._pipeline = pipeline
        tcp_source = _make_element(self._runtime, "tcpclientsrc", "tcp-client")
        depayloader = _make_element(self._runtime, "gdpdepay", "packet-depayloader")
        self._appsink = _make_element(self._runtime, "appsink", "packet-sink")
        tcp_source.set_property("host", self._host)
        tcp_source.set_property("port", self._port)
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)
        self._appsink.connect("new-sample", self._on_new_sample)
        _add_and_link(self._pipeline, (tcp_source, depayloader, self._appsink))

        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_bus_error)
        self._bus.connect("message::eos", self._on_bus_eos)

    @property
    def received_count(self) -> int:
        with self._condition:
            return self._received_count

    @property
    def failure(self) -> RuntimeError | None:
        with self._condition:
            return self._failure

    def _record_failure(self, failure: RuntimeError) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = failure
            self._condition.notify_all()
        self._loop.quit()

    def _on_bus_error(self, _bus: Any, message: Any) -> None:
        error, debug = message.parse_error()
        detail = f": {debug}" if debug else ""
        self._record_failure(RuntimeError(f"GStreamer receiver error: {error}{detail}"))

    def _on_bus_eos(self, _bus: Any, _message: Any) -> None:
        with self._condition:
            self._condition.notify_all()
        self._loop.quit()

    def _on_new_sample(self, appsink: Any) -> Any:
        Gst = self._runtime.Gst
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        mapped, map_info = buffer.map(Gst.MapFlags.READ)
        if not mapped:
            self._record_failure(
                RuntimeError("GStreamer could not map a received packet")
            )
            return Gst.FlowReturn.ERROR

        try:
            packet_bytes = bytes(map_info.data)
            self._consumer.consume_packet_bytes(packet_bytes)
        except Exception as exc:
            self._record_failure(
                RuntimeError(f"GStreamer packet consumer failed: {exc}")
            )
            return Gst.FlowReturn.ERROR
        finally:
            buffer.unmap(map_info)

        with self._condition:
            self._received_count += 1
            self._condition.notify_all()
        return Gst.FlowReturn.OK

    def _run_loop(self) -> None:
        Gst = self._runtime.Gst
        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._record_failure(
                RuntimeError(
                    "GStreamer receiver pipeline failed to enter PLAYING state"
                )
            )
        self._started.set()
        if self.failure is None:
            self._loop.run()

    def start(self, *, timeout_s: float = 5.0) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if self._closed:
            raise RuntimeError("GStreamer receiver is closed")
        if self._thread.is_alive() or self._started.is_set():
            raise RuntimeError("GStreamer receiver has already been started")

        self._thread.start()
        if not self._started.wait(timeout_s):
            self.close()
            raise TimeoutError("GStreamer receiver did not start before the timeout")
        if self.failure is not None:
            raise self.failure

    def wait_for_packets(self, count: int = 1, *, timeout_s: float = 5.0) -> bool:
        if count <= 0:
            raise ValueError("count must be positive")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

        deadline = monotonic() + timeout_s
        with self._condition:
            while self._received_count < count and self._failure is None:
                remaining = deadline - monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if self._failure is not None:
                raise self._failure
            return self._received_count >= count

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.quit()
        self._pipeline.set_state(self._runtime.Gst.State.NULL)
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._bus.remove_signal_watch()

    def __enter__(self) -> GStreamerPacketReceiver:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
