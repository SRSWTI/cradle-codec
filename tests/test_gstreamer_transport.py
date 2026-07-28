import socket
import unittest

from cradle_codec.gstreamer import (
    GStreamerPacketReceiver,
    GStreamerPacketSender,
    GStreamerUnavailableError,
    gstreamer_available,
    require_gstreamer,
)


class RecordingPacketConsumer:
    def __init__(self) -> None:
        self.packets: list[bytes] = []

    def consume_packet_bytes(self, packet_bytes: bytes) -> None:
        self.packets.append(packet_bytes)


def available_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class GStreamerTransportConfigurationTests(unittest.TestCase):
    def test_runtime_probe_has_explicit_unavailable_error(self) -> None:
        if gstreamer_available():
            self.assertIsNotNone(require_gstreamer().Gst)
        else:
            with self.assertRaises(GStreamerUnavailableError):
                require_gstreamer()

    def test_sender_rejects_invalid_port_before_loading_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
            GStreamerPacketSender(port=0)

    def test_receiver_requires_packet_consumer_before_loading_runtime(self) -> None:
        with self.assertRaisesRegex(TypeError, "consume_packet_bytes"):
            GStreamerPacketReceiver("127.0.0.1", 42_425, object())


@unittest.skipUnless(
    gstreamer_available(), "GStreamer PyGObject runtime is unavailable"
)
class GStreamerTransportLoopbackTests(unittest.TestCase):
    def test_packet_round_trip_preserves_bytes(self) -> None:
        port = available_tcp_port()
        expected = b"cradle-codec-gstreamer\x00packet"
        consumer = RecordingPacketConsumer()
        sender = GStreamerPacketSender(bind_host="127.0.0.1", port=port)
        receiver = GStreamerPacketReceiver("127.0.0.1", port, consumer)
        try:
            receiver.start()
            self.assertTrue(sender.wait_for_clients(), "receiver did not connect")
            result = sender.send(expected)
            self.assertTrue(result.accepted, result.flow_return)
            self.assertTrue(receiver.wait_for_packets(), "packet was not delivered")
            self.assertEqual(consumer.packets, [expected])
        finally:
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
