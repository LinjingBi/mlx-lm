"""Lightweight Python client for the local MLX-LM Unix-socket daemon."""

import socket
import uuid
from typing import Iterator, Optional, Union

from .paths import DEFAULT_SOCKET_PATH
from .protocol import (
    ProtocolError,
    recv_frame,
    request_frame,
    send_frame,
    validate_response_frame,
)
from .types import (
    GenerateRequest,
    GenerationDelta,
    GenerationFinished,
    GenerationResult,
    GenerationStarted,
)


ClientEvent = Union[GenerationStarted, GenerationDelta, GenerationFinished]


class RuntimeRemoteError(RuntimeError):
    def __init__(self, message: str, code: str = "runtime_error"):
        super().__init__(message)
        self.code = code


class UnixRuntimeClient:
    """Connect short-lived callers to a resident local runtime daemon."""

    def __init__(self, socket_path=None, *, timeout: Optional[float] = None):
        self.socket_path = str(socket_path or DEFAULT_SOCKET_PATH)
        self.timeout = timeout

    def _connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
        except Exception:
            sock.close()
            raise
        return sock

    def _request(self, operation, data=None):
        request_id = uuid.uuid4().hex
        with self._connect() as sock:
            send_frame(sock, request_frame(request_id, operation, data))
            response = recv_frame(sock)
        if response is None:
            raise ProtocolError("Runtime closed the connection without a response.")
        frame_type = validate_response_frame(response, request_id)
        if frame_type == "error":
            raise RuntimeRemoteError(
                response.get("message", "Runtime request failed."),
                response.get("code", "runtime_error"),
            )
        if frame_type != "result":
            raise ProtocolError(
                "Expected a result frame, received {!r}.".format(frame_type)
            )
        data = response.get("data", {})
        if not isinstance(data, dict):
            raise ProtocolError("Runtime result data must be an object.")
        return data

    def health(self):
        return self._request("health")

    def status(self):
        return self._request("status")

    def clear_cache(self):
        return self._request("clear_cache")

    def unload(self):
        return self._request("unload")

    def shutdown(self):
        return self._request("shutdown")

    def stream_events(self, request: GenerateRequest) -> Iterator[ClientEvent]:
        request.validate()
        request_id = uuid.uuid4().hex
        with self._connect() as sock:
            send_frame(
                sock,
                request_frame(request_id, "generate", request.to_dict()),
            )
            while True:
                response = recv_frame(sock)
                if response is None:
                    raise ProtocolError(
                        "Runtime closed the connection before generation finished."
                    )
                frame_type = validate_response_frame(response, request_id)
                if frame_type == "error":
                    raise RuntimeRemoteError(
                        response.get("message", "Generation failed."),
                        response.get("code", "runtime_error"),
                    )
                if frame_type == "started":
                    yield GenerationStarted(
                        model=response["model"],
                        prompt_tokens=response["prompt_tokens"],
                        cached_tokens=response["cached_tokens"],
                    )
                elif frame_type == "delta":
                    yield GenerationDelta(
                        text=response.get("text", ""),
                        token=response["token"],
                        finish_reason=response.get("finish_reason"),
                    )
                elif frame_type == "finished":
                    yield GenerationFinished(
                        finish_reason=response["finish_reason"],
                        prompt_tokens=response["prompt_tokens"],
                        cached_tokens=response["cached_tokens"],
                        generation_tokens=response["generation_tokens"],
                        prompt_tps=response["prompt_tps"],
                        generation_tps=response["generation_tps"],
                        peak_memory_gb=response["peak_memory_gb"],
                        ttft_seconds=response["ttft_seconds"],
                        total_seconds=response["total_seconds"],
                    )
                    return
                else:
                    raise ProtocolError(
                        "Unknown generation frame type {!r}.".format(frame_type)
                    )

    def stream_generate(
        self, request: Optional[GenerateRequest] = None, **kwargs
    ) -> Iterator[GenerationDelta]:
        if request is not None and kwargs:
            raise ValueError("Pass either a GenerateRequest or keyword arguments.")
        request = request or GenerateRequest(**kwargs)
        for event in self.stream_events(request):
            if isinstance(event, GenerationDelta):
                yield event

    def generate(self, request: Optional[GenerateRequest] = None, **kwargs):
        if request is not None and kwargs:
            raise ValueError("Pass either a GenerateRequest or keyword arguments.")
        request = request or GenerateRequest(**kwargs)
        started = None
        finished = None
        text = []
        for event in self.stream_events(request):
            if isinstance(event, GenerationStarted):
                started = event
            elif isinstance(event, GenerationDelta):
                text.append(event.text)
            elif isinstance(event, GenerationFinished):
                finished = event
        if started is None or finished is None:
            raise ProtocolError("Generation did not return complete metadata.")
        return GenerationResult("".join(text), started, finished)
