"""Length-prefixed JSON protocol used by the local Unix-socket runtime."""

import json
import struct
from typing import Any, Dict, Optional


PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_SIZE = 8 * 1024 * 1024
_LENGTH = struct.Struct("!I")


class ProtocolError(Exception):
    """A peer sent a malformed or incompatible protocol frame."""


class ConnectionClosed(EOFError):
    """The peer closed the socket before a complete frame was received."""


def encode_frame(payload: Dict[str, Any], max_size: int = DEFAULT_MAX_FRAME_SIZE):
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol frames must be JSON objects.")
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Protocol frame is not JSON serializable.") from exc
    if len(body) > max_size:
        raise ProtocolError(
            "Protocol frame is too large: {} bytes (limit {}).".format(
                len(body), max_size
            )
        )
    return _LENGTH.pack(len(body)) + body


def _recv_exact(sock, size: int, allow_clean_eof: bool = False):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if allow_clean_eof and remaining == size:
                return None
            raise ConnectionClosed("Connection closed in the middle of a frame.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock, max_size: int = DEFAULT_MAX_FRAME_SIZE):
    header = _recv_exact(sock, _LENGTH.size, allow_clean_eof=True)
    if header is None:
        return None
    (size,) = _LENGTH.unpack(header)
    if size > max_size:
        raise ProtocolError(
            "Protocol frame declares {} bytes (limit {}).".format(size, max_size)
        )
    body = _recv_exact(sock, size)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Protocol frame contains invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol frames must decode to JSON objects.")
    return payload


def send_frame(sock, payload: Dict[str, Any], max_size: int = DEFAULT_MAX_FRAME_SIZE):
    sock.sendall(encode_frame(payload, max_size=max_size))


def request_frame(
    request_id: str, operation: str, data: Optional[Dict[str, Any]] = None
):
    return {
        "version": PROTOCOL_VERSION,
        "id": request_id,
        "operation": operation,
        "data": data or {},
    }


def response_frame(request_id: str, frame_type: str, **data):
    return {
        "version": PROTOCOL_VERSION,
        "id": request_id,
        "type": frame_type,
        **data,
    }


def validate_request_frame(payload: Dict[str, Any]):
    if payload.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            "Unsupported protocol version {!r}; expected {}.".format(
                payload.get("version"), PROTOCOL_VERSION
            )
        )
    request_id = payload.get("id")
    operation = payload.get("operation")
    data = payload.get("data", {})
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("Request 'id' must be a non-empty string.")
    if not isinstance(operation, str) or not operation:
        raise ProtocolError("Request 'operation' must be a non-empty string.")
    if not isinstance(data, dict):
        raise ProtocolError("Request 'data' must be an object.")
    return request_id, operation, data


def validate_response_frame(payload: Dict[str, Any], request_id: str):
    if payload.get("version") != PROTOCOL_VERSION:
        raise ProtocolError("Runtime returned an incompatible protocol version.")
    if payload.get("id") != request_id:
        raise ProtocolError("Runtime returned a response for another request.")
    frame_type = payload.get("type")
    if not isinstance(frame_type, str):
        raise ProtocolError("Response 'type' must be a string.")
    return frame_type
