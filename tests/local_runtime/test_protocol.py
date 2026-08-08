import json
import socket
import struct
import unittest

from mlx_lm_runtime.protocol import (
    ConnectionClosed,
    ProtocolError,
    encode_frame,
    recv_frame,
)
from mlx_lm_runtime.types import GenerateRequest


class TestLocalRuntimeProtocol(unittest.TestCase):
    def test_fragmented_frame(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        encoded = encode_frame({"message": "hello", "number": 7})
        for byte in encoded:
            left.sendall(bytes([byte]))
        self.assertEqual(recv_frame(right), {"message": "hello", "number": 7})

    def test_clean_eof_and_partial_frame_differ(self):
        left, right = socket.socketpair()
        left.close()
        self.assertIsNone(recv_frame(right))
        right.close()

        left, right = socket.socketpair()
        self.addCleanup(right.close)
        left.sendall(b"\x00\x00")
        left.close()
        with self.assertRaises(ConnectionClosed):
            recv_frame(right)

    def test_invalid_json_and_oversized_frame(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        body = b"not-json"
        left.sendall(struct.pack("!I", len(body)) + body)
        with self.assertRaises(ProtocolError):
            recv_frame(right)

        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        left.sendall(struct.pack("!I", 100))
        with self.assertRaises(ProtocolError):
            recv_frame(right, max_size=10)

    def test_generate_request_round_trip_and_validation(self):
        request = GenerateRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=12,
            stop=("\n",),
        )
        self.assertEqual(
            GenerateRequest.from_dict(json.loads(json.dumps(request.to_dict()))),
            request,
        )
        with self.assertRaises(ValueError):
            GenerateRequest(prompt="a", messages=[]).validate()
        with self.assertRaises(ValueError):
            GenerateRequest.from_dict({"prompt": "a", "unsupported": True})


if __name__ == "__main__":
    unittest.main()
