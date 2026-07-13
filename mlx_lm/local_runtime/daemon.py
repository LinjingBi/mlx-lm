"""Sequential Unix-domain-socket daemon for a resident MLX-LM model."""

import dataclasses
import errno
import logging
import os
import signal
import socket
from pathlib import Path

from .errors import RuntimeBusy
from mlx_lm_runtime.paths import DEFAULT_SOCKET_PATH
from mlx_lm_runtime.protocol import (
    ConnectionClosed,
    ProtocolError,
    recv_frame,
    response_frame,
    send_frame,
    validate_request_frame,
)
from mlx_lm_runtime.types import (
    GenerateRequest,
    GenerationDelta,
    GenerationFinished,
    GenerationStarted,
)


class RuntimeDaemon:
    """Serve one request at a time and never execute concurrent inference."""

    def __init__(self, engine, socket_path=None):
        self.engine = engine
        self.socket_path = Path(socket_path or DEFAULT_SOCKET_PATH).expanduser()
        self._listener = None
        self._stopping = False

    def _prepare_socket(self):
        parent_existed = self.socket_path.parent.exists()
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        default_parent = Path(DEFAULT_SOCKET_PATH).expanduser().parent
        if not parent_existed or self.socket_path.parent == default_parent:
            os.chmod(self.socket_path.parent, 0o700)
        if self.socket_path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.socket_path))
            except OSError as exc:
                if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                    raise
                self.socket_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "A runtime is already listening at {}.".format(self.socket_path)
                )
            finally:
                probe.close()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
        except Exception:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener

    def _stop(self, *_):
        self._stopping = True
        if self._listener is not None:
            self._listener.close()

    def _send_error(self, connection, request_id, exc):
        code = "busy" if isinstance(exc, RuntimeBusy) else "runtime_error"
        if isinstance(exc, (ValueError, ProtocolError)):
            code = "invalid_request"
        send_frame(
            connection,
            response_frame(
                request_id or "unknown",
                "error",
                code=code,
                message=str(exc),
            ),
        )

    def _handle_generate(self, connection, request_id, data):
        request = GenerateRequest.from_dict(data)
        events = self.engine.stream(request)
        try:
            for event in events:
                payload = dataclasses.asdict(event)
                if isinstance(event, GenerationStarted):
                    frame_type = "started"
                elif isinstance(event, GenerationDelta):
                    frame_type = "delta"
                elif isinstance(event, GenerationFinished):
                    frame_type = "finished"
                else:
                    raise RuntimeError(
                        "Engine returned unsupported event {!r}.".format(event)
                    )
                send_frame(
                    connection,
                    response_frame(request_id, frame_type, **payload),
                )
        finally:
            events.close()

    def _handle_connection(self, connection):
        request_id = None
        try:
            payload = recv_frame(connection)
            if payload is None:
                return
            request_id, operation, data = validate_request_frame(payload)
            if operation == "generate":
                self._handle_generate(connection, request_id, data)
            elif operation == "health":
                send_frame(
                    connection,
                    response_frame(
                        request_id,
                        "result",
                        data={"status": "ok", "sequential": True},
                    ),
                )
            elif operation == "status":
                send_frame(
                    connection,
                    response_frame(request_id, "result", data=self.engine.status()),
                )
            elif operation == "clear_cache":
                self.engine.clear_cache()
                send_frame(
                    connection,
                    response_frame(request_id, "result", data={"cache_cleared": True}),
                )
            elif operation == "shutdown":
                send_frame(
                    connection,
                    response_frame(request_id, "result", data={"shutting_down": True}),
                )
                self._stop()
            else:
                raise ValueError("Unknown operation {!r}.".format(operation))
        except (BrokenPipeError, ConnectionResetError, ConnectionClosed):
            logging.info("Local runtime client disconnected.")
        except Exception as exc:
            logging.exception("Local runtime request failed.")
            try:
                self._send_error(connection, request_id, exc)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    def serve_forever(self, *, install_signal_handlers: bool = True):
        self._prepare_socket()
        if install_signal_handlers:
            signal.signal(signal.SIGTERM, self._stop)
            signal.signal(signal.SIGINT, self._stop)
        logging.info("Local runtime listening on %s", self.socket_path)
        try:
            while not self._stopping:
                try:
                    connection, _ = self._listener.accept()
                except OSError as exc:
                    if self._stopping or exc.errno in (errno.EBADF, errno.EINVAL):
                        break
                    raise
                with connection:
                    self._handle_connection(connection)
        finally:
            if self._listener is not None:
                self._listener.close()
                self._listener = None
            self.socket_path.unlink(missing_ok=True)

    def close(self):
        self._stop()
