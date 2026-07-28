from __future__ import annotations

import json
import threading
from dataclasses import asdict

import typer

from cradle_codec.gstreamer import (
    GStreamerPacketReceiver,
    GStreamerPacketSender,
    GStreamerUnavailableError,
)

app = typer.Typer(
    help="Run the optional GStreamer packet transport.", no_args_is_help=True
)


class _SinglePacketConsumer:
    def __init__(self) -> None:
        self.packet: bytes | None = None
        self.received = threading.Event()

    def consume_packet_bytes(self, packet_bytes: bytes) -> None:
        self.packet = packet_bytes
        self.received.set()


@app.command("loopback")
def loopback_command(
    payload: str = typer.Option(
        "cradle codec gstreamer loopback",
        "--payload",
        help="UTF-8 payload sent through GDP over TCP.",
    ),
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Local address used by both endpoints."
    ),
    port: int = typer.Option(42_425, "--port", min=1, max=65_535, help="TCP port."),
    timeout_s: float = typer.Option(
        5.0, "--timeout-s", min=0.001, help="Startup and receive timeout."
    ),
) -> None:
    """Send one packet through a local GStreamer sender and receiver."""

    expected = payload.encode("utf-8")
    consumer = _SinglePacketConsumer()
    sender: GStreamerPacketSender | None = None
    receiver: GStreamerPacketReceiver | None = None
    try:
        sender = GStreamerPacketSender(bind_host=host, port=port)
        receiver = GStreamerPacketReceiver(host, port, consumer)
        receiver.start(timeout_s=timeout_s)
        if not sender.wait_for_clients(timeout_s=timeout_s):
            raise TimeoutError(
                "GStreamer loopback receiver did not connect before the timeout"
            )
        result = sender.send(expected)
        if not result.accepted:
            raise RuntimeError(f"GStreamer rejected the packet: {result.flow_return}")
        if not receiver.wait_for_packets(timeout_s=timeout_s):
            raise TimeoutError(
                "GStreamer loopback did not receive a packet before the timeout"
            )
        if consumer.packet != expected:
            raise RuntimeError("GStreamer loopback payload mismatch")
    except (GStreamerUnavailableError, RuntimeError, TimeoutError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        if sender is not None:
            sender.close()
        if receiver is not None:
            receiver.close()

    typer.echo(
        json.dumps(
            {
                **asdict(result),
                "port": port,
                "received_bytes": len(expected),
            },
            sort_keys=True,
        )
    )
