import time
import threading
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.kv_gstreamer_sender import GstreamerLayerPacketSender
from kv_cache_store.kv_gstreamer_receiver import GstreamerLayerPacketReceiver

class MockConsumer:
    def __init__(self):
        self.received = []
    
    def consume_packet_bytes(self, packet_bytes: bytes):
        print(f"Consumer received {len(packet_bytes)} bytes!")
        self.received.append(packet_bytes)

def main():
    print("Starting GStreamer test...")
    consumer = MockConsumer()
    sender = GstreamerLayerPacketSender(port=42425)
    time.sleep(1)
    
    receiver = GstreamerLayerPacketReceiver("127.0.0.1", 42425, consumer)
    receiver.start()
    time.sleep(1)
    
    # Send some data
    test_data = b"Hello, GStreamer KV Streaming!" * 10
    print(f"Sender sending {len(test_data)} bytes...")
    res = sender.send(test_data)
    print(f"Sender result: {res}")
    
    # Wait for data to arrive
    time.sleep(2)
    
    sender.shutdown()
    receiver.shutdown()
    
    if len(consumer.received) > 0 and consumer.received[0] == test_data:
        print("SUCCESS! Data matched perfectly.")
    else:
        print("FAILED! Data did not arrive or did not match.")
        
if __name__ == "__main__":
    main()
