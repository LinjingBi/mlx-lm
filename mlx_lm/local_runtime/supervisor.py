"""Concurrent socket supervisor for a disposable MLX model worker."""

import atexit
import datetime
import errno
import logging
import multiprocessing
import os
import signal
import socket
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

from .errors import RuntimeBusy
from .worker import run_worker
from mlx_lm_runtime.paths import DEFAULT_SOCKET_PATH
from mlx_lm_runtime.protocol import (
    ConnectionClosed,
    ProtocolError,
    recv_frame,
    response_frame,
    send_frame,
    validate_request_frame,
)


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


class WorkerManager:
    """Own one disposable worker and multiplex concurrent generate streams."""

    def __init__(
        self,
        config,
        *,
        idle_timeout=300.0,
        startup_timeout=600.0,
        stop_timeout=5.0,
        process_context=None,
        worker_target=run_worker,
    ):
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be zero or greater.")
        self.config = dict(config)
        self.idle_timeout = float(idle_timeout)
        self.startup_timeout = float(startup_timeout)
        self.stop_timeout = float(stop_timeout)
        self.decode_concurrency = int(self.config.get("decode_concurrency", 32))
        if self.decode_concurrency <= 0:
            raise ValueError("decode_concurrency must be greater than zero.")
        self._context = process_context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._pipe_lock = threading.Lock()
        self._process = None
        self._connection = None
        self._idle_timer = None
        self._closed = False
        self._state = "unloaded"
        self._unload_reason = "never_loaded"
        self._worker_status = {}
        self._loading_started = None
        self._request_started = None
        self._last_activity_monotonic = None
        self._last_activity_at = None
        self._unloaded_at = None
        self._last_model_load_ms = None
        self._estimated_cold_start_ms = None
        self._estimated_warm_first_token_ms = None
        self._cold_start_count = 0
        self._worker_restart_count = 0
        self._active_count = 0
        self._streams = {}
        self._control_queue = Queue()
        self._reader_stop = threading.Event()
        self._reader_thread = None
        # Ensure workers never block interpreter exit if a test or crash
        # skips an explicit close(); multiprocessing joins non-daemons forever.
        atexit.register(self._atexit_close)

    def _set_state(self, state, **updates):
        with self._state_lock:
            self._state = state
            for name, value in updates.items():
                setattr(self, name, value)

    def _cancel_idle_timer(self):
        with self._state_lock:
            timer = self._idle_timer
            self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_idle_timer(self):
        self._cancel_idle_timer()
        if self.idle_timeout == 0 or self._closed:
            return
        timer = threading.Timer(self.idle_timeout, self._idle_expired)
        timer.daemon = True
        with self._state_lock:
            if self._closed or self._state != "ready" or self._active_count != 0:
                return
            self._idle_timer = timer
        timer.start()

    def _idle_expired(self):
        if self._closed:
            return
        if not self._lifecycle_lock.acquire(blocking=False):
            if not self._closed:
                self._schedule_idle_timer()
            return
        try:
            with self._state_lock:
                if self._closed or self._state != "ready" or self._active_count != 0:
                    return
            self._stop_worker_locked("idle_timeout")
        finally:
            self._lifecycle_lock.release()

    def _send(self, message):
        with self._pipe_lock:
            if self._connection is None:
                raise RuntimeError("Model worker is not connected.")
            self._connection.send(message)

    def _reader_loop(self):
        connection = self._connection
        while not self._reader_stop.is_set():
            try:
                if connection is None or not connection.poll(0.05):
                    continue
                message = connection.recv()
            except (EOFError, BrokenPipeError, OSError):
                self._fail_all_streams("Model worker exited unexpectedly.")
                return
            kind = message.get("kind")
            request_id = message.get("id")
            if kind in ("event", "complete", "error") and request_id:
                with self._state_lock:
                    queue = self._streams.get(request_id)
                if queue is not None:
                    queue.put(message)
                continue
            if kind in ("cleared", "stopping", "error"):
                self._control_queue.put(message)
                continue

    def _fail_all_streams(self, message):
        with self._state_lock:
            streams = list(self._streams.values())
        for queue in streams:
            queue.put({"kind": "error", "message": message})

    def _start_reader(self):
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="mlx-lm-runtime-worker-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_reader(self):
        self._reader_stop.set()
        thread = self._reader_thread
        self._reader_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.stop_timeout)

    def _start_worker_locked(self):
        if self._closed:
            raise RuntimeError("Worker manager is closed.")
        with self._state_lock:
            if self._state in ("ready", "busy"):
                return
            previous_reason = self._unload_reason
            self._state = "loading"
            self._loading_started = time.monotonic()
            self._unload_reason = None
        if self._process is not None:
            self._stop_process()
        logging.info("event=worker_starting reason=cold_request")
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=self._worker_target,
            args=(child, self.config),
            name="mlx-lm-runtime-worker",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(self.startup_timeout):
            self._stop_process()
            self._set_state(
                "failed",
                _unload_reason="startup_timeout",
                _loading_started=None,
            )
            raise RuntimeError("Model worker did not start before the timeout.")
        message = parent.recv()
        if message.get("kind") != "ready":
            self._stop_process()
            self._set_state(
                "failed",
                _unload_reason="startup_failure",
                _loading_started=None,
            )
            detail = message.get("message", "Model worker failed during startup.")
            raise RuntimeError(detail)
        load_ms = message["load_ms"]
        with self._state_lock:
            self._state = "ready"
            self._loading_started = None
            self._last_model_load_ms = load_ms
            if self._estimated_cold_start_ms is None:
                self._estimated_cold_start_ms = load_ms
            else:
                self._estimated_cold_start_ms = round(
                    0.7 * self._estimated_cold_start_ms + 0.3 * load_ms, 3
                )
            self._cold_start_count += 1
            if previous_reason not in (None, "never_loaded", "idle_timeout"):
                self._worker_restart_count += 1
            self._worker_status = message.get("status", {})
        self._start_reader()
        logging.info("event=model_loaded duration_ms=%s pid=%s", load_ms, process.pid)

    def _stop_process(self):
        self._stop_reader()
        process = self._process
        connection = self._connection
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        process.join(timeout=self.stop_timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self.stop_timeout)
        if process.is_alive():
            if hasattr(process, "kill"):
                process.kill()
            elif process.pid is not None:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except OSError:
                    pass
            process.join(timeout=self.stop_timeout)

    def _stop_worker_locked(self, reason):
        self._cancel_idle_timer()
        with self._state_lock:
            if self._process is None:
                self._state = "unloaded"
                self._unload_reason = reason
                return
            self._state = "unloading"
            cache_bytes = self._worker_status.get("prompt_cache_bytes", 0)
            connection = self._connection
        logging.info(
            "event=worker_unloading reason=%s cache_bytes=%s", reason, cache_bytes
        )
        if connection is not None:
            try:
                self._send({"operation": "stop"})
                try:
                    message = self._control_queue.get(timeout=self.stop_timeout)
                except Empty:
                    message = None
                if message is None and connection.poll(self.stop_timeout):
                    connection.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        self._stop_process()
        with self._state_lock:
            self._state = "unloaded"
            self._unload_reason = reason
            self._unloaded_at = _now_iso()
            self._worker_status = {}
            self._active_count = 0
            self._streams.clear()
        logging.info("event=worker_stopped reason=%s", reason)

    def stream(self, request_data):
        request_id = uuid.uuid4().hex
        queue = Queue()
        cold_start = False
        reserved = False
        started = False
        finished = None

        with self._state_lock:
            if self._active_count >= self.decode_concurrency:
                raise RuntimeBusy(
                    "The local runtime is at decode concurrency cap ({}).".format(
                        self.decode_concurrency
                    )
                )
            self._active_count += 1
            reserved = True
            self._streams[request_id] = queue
            cold_start = self._state not in ("ready", "busy")

        self._cancel_idle_timer()
        try:
            with self._lifecycle_lock:
                self._start_worker_locked()
            with self._state_lock:
                self._state = "busy"
                if self._request_started is None:
                    self._request_started = time.monotonic()
            logging.info(
                "event=generation_started id=%s cold_start=%s active=%s",
                request_id,
                cold_start,
                self._active_count,
            )
            self._send(
                {
                    "operation": "generate",
                    "id": request_id,
                    "data": request_data,
                }
            )
            started = True

            while True:
                message = queue.get()
                kind = message.get("kind")
                if kind == "event":
                    if message["event"] == "GenerationFinished":
                        finished = message
                    else:
                        yield message
                elif kind == "complete":
                    with self._state_lock:
                        self._worker_status = message.get("status", {})
                    if finished is not None:
                        yield finished
                    break
                elif kind == "error":
                    code = message.get("code")
                    text = message.get("message", "Worker request failed.")
                    if code == "busy":
                        raise RuntimeBusy(text)
                    raise RuntimeError(text)
                else:
                    raise RuntimeError("Unexpected worker response {!r}.".format(kind))

            finished_data = (finished or {}).get("data", {})
            warm_ttft_ms = finished_data.get("ttft_seconds")
            if warm_ttft_ms is not None and not cold_start:
                warm_ttft_ms *= 1000
                with self._state_lock:
                    if self._estimated_warm_first_token_ms is None:
                        self._estimated_warm_first_token_ms = warm_ttft_ms
                    else:
                        self._estimated_warm_first_token_ms = round(
                            0.7 * self._estimated_warm_first_token_ms
                            + 0.3 * warm_ttft_ms,
                            3,
                        )
            logging.info(
                "event=generation_finished id=%s cold_start=%s total_seconds=%s",
                request_id,
                cold_start,
                finished_data.get("total_seconds"),
            )
        except GeneratorExit:
            if started:
                try:
                    self._send({"operation": "cancel", "id": request_id})
                except Exception:
                    pass
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._stop_process()
            self._set_state(
                "failed",
                _unload_reason="worker_failure",
                _request_started=None,
            )
            logging.exception("event=worker_failed")
            raise RuntimeError("Model worker exited unexpectedly.") from exc
        finally:
            with self._state_lock:
                self._streams.pop(request_id, None)
                if reserved:
                    self._active_count = max(0, self._active_count - 1)
                active_count = self._active_count
                if active_count == 0 and self._state == "busy":
                    self._state = "ready"
                    self._request_started = None
                    self._last_activity_monotonic = time.monotonic()
                    self._last_activity_at = _now_iso()
                ready = self._state == "ready" and active_count == 0
            if ready and not self._closed:
                self._schedule_idle_timer()

    def clear_cache(self):
        with self._state_lock:
            if self._active_count > 0:
                raise RuntimeBusy("Cannot clear the cache during generation.")
            state = self._state
        if state == "unloaded":
            return
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeBusy("Cannot clear the cache while the worker is changing.")
        try:
            with self._state_lock:
                if self._active_count > 0:
                    raise RuntimeBusy("Cannot clear the cache during generation.")
                if self._state != "ready":
                    raise RuntimeBusy(
                        "Cannot clear the cache while worker is {}.".format(self._state)
                    )
            # Drop stale control messages.
            while True:
                try:
                    self._control_queue.get_nowait()
                except Empty:
                    break
            self._send({"operation": "clear_cache"})
            message = self._control_queue.get(timeout=self.stop_timeout)
            if message.get("kind") == "error":
                if message.get("code") == "busy":
                    raise RuntimeBusy(message.get("message", "Worker is busy."))
                raise RuntimeError(
                    message.get("message", "Could not clear worker cache.")
                )
            if message.get("kind") != "cleared":
                raise RuntimeError(
                    message.get("message", "Could not clear worker cache.")
                )
            with self._state_lock:
                self._worker_status = message.get("status", {})
        finally:
            self._lifecycle_lock.release()

    def unload(self):
        with self._state_lock:
            if self._active_count > 0:
                raise RuntimeBusy("Cannot unload the model during generation.")
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeBusy("Cannot unload the model while the worker is changing.")
        try:
            with self._state_lock:
                if self._active_count > 0:
                    raise RuntimeBusy("Cannot unload the model during generation.")
            self._stop_worker_locked("explicit_unload")
        finally:
            self._lifecycle_lock.release()

    def close(self):
        self._closed = True
        self._cancel_idle_timer()
        self._lifecycle_lock.acquire()
        try:
            self._stop_worker_locked("supervisor_shutdown")
        finally:
            self._lifecycle_lock.release()

    def _atexit_close(self):
        if self._closed and self._process is None:
            return
        try:
            self.close()
        except Exception:
            process = self._process
            if process is not None and process.is_alive():
                try:
                    process.kill()
                except Exception:
                    pass

    def health(self):
        with self._state_lock:
            return {
                "status": "ok",
                "sequential": False,
                "model_ready": self._state in ("ready", "busy"),
                "active_requests": self._active_count,
                "decode_concurrency": self.decode_concurrency,
            }

    def status(self):
        now = time.monotonic()
        with self._state_lock:
            state = self._state
            process = self._process
            worker_status = dict(self._worker_status)
            active_count = self._active_count
            idle_for = (
                now - self._last_activity_monotonic
                if self._last_activity_monotonic is not None
                else None
            )
            status = {
                "service_state": "running",
                "worker_state": state,
                "model": self.config["model"],
                "adapter": self.config.get("adapter_path"),
                "model_resident": state in ("ready", "busy", "unloading"),
                "worker_pid": process.pid if process is not None else None,
                "next_request_cold": state
                in ("unloaded", "loading", "unloading", "failed"),
                "idle_timeout_seconds": self.idle_timeout,
                "idle_for_seconds": round(idle_for, 3)
                if idle_for is not None
                else None,
                "unload_in_seconds": (
                    round(max(0.0, self.idle_timeout - idle_for), 3)
                    if idle_for is not None
                    and self.idle_timeout > 0
                    and state == "ready"
                    and active_count == 0
                    else None
                ),
                "unload_reason": self._unload_reason,
                "loading_for_ms": (
                    round((now - self._loading_started) * 1000, 3)
                    if self._loading_started is not None
                    else None
                ),
                "request_running_for_ms": (
                    round((now - self._request_started) * 1000, 3)
                    if self._request_started is not None
                    else None
                ),
                "last_model_load_ms": self._last_model_load_ms,
                "estimated_cold_start_ms": self._estimated_cold_start_ms,
                "estimated_warm_first_token_ms": self._estimated_warm_first_token_ms,
                "cold_start_count": self._cold_start_count,
                "worker_restart_count": self._worker_restart_count,
                "last_activity_at": self._last_activity_at,
                "unloaded_at": self._unloaded_at,
                "active": active_count > 0,
                "active_requests": active_count,
                "decode_concurrency": self.decode_concurrency,
                "prompt_concurrency": self.config.get("prompt_concurrency", 8),
                "prefill_step_size": self.config.get("prefill_step_size", 2048),
                "prompt_cache_entries": 0,
                "prompt_cache_bytes": 0,
                "prompt_cache_by_type": {
                    cache_type: {"n_sequences": 0, "n_bytes": 0}
                    for cache_type in ("assistant", "user", "system")
                },
            }
        status.update(worker_status)
        status["active"] = active_count > 0
        status["active_requests"] = active_count
        return status


class RuntimeSupervisor:
    """Keep the public socket alive while model workers come and go."""

    def __init__(self, manager, socket_path=None):
        self.manager = manager
        self.socket_path = Path(socket_path or DEFAULT_SOCKET_PATH).expanduser()
        self._listener = None
        self._stopping = threading.Event()
        self._threads = set()
        self._threads_lock = threading.Lock()

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
            listener.listen(16)
        except Exception:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener

    def _stop(self, *_):
        self._stopping.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

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

    def _handle_connection(self, connection):
        request_id = None
        stream = None
        try:
            payload = recv_frame(connection)
            if payload is None:
                return
            request_id, operation, data = validate_request_frame(payload)
            if operation == "generate":
                stream = self.manager.stream(data)
                for message in stream:
                    event_names = {
                        "GenerationStarted": "started",
                        "GenerationDelta": "delta",
                        "GenerationFinished": "finished",
                    }
                    send_frame(
                        connection,
                        response_frame(
                            request_id,
                            event_names[message["event"]],
                            **message["data"],
                        ),
                    )
            elif operation == "health":
                send_frame(
                    connection,
                    response_frame(request_id, "result", data=self.manager.health()),
                )
            elif operation == "status":
                send_frame(
                    connection,
                    response_frame(request_id, "result", data=self.manager.status()),
                )
            elif operation == "clear_cache":
                self.manager.clear_cache()
                send_frame(
                    connection,
                    response_frame(request_id, "result", data={"cache_cleared": True}),
                )
            elif operation == "unload":
                self.manager.unload()
                send_frame(
                    connection,
                    response_frame(request_id, "result", data={"unloaded": True}),
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
            if stream is not None:
                stream.close()
        except Exception as exc:
            logging.exception("Local runtime request failed.")
            try:
                self._send_error(connection, request_id, exc)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        finally:
            connection.close()
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def serve_forever(self, *, install_signal_handlers=True):
        self._prepare_socket()
        # Poll so shutdown from a handler thread does not rely on close()
        # unblocking accept(), which is not reliable across platforms.
        self._listener.settimeout(0.5)
        if install_signal_handlers:
            signal.signal(signal.SIGTERM, self._stop)
            signal.signal(signal.SIGINT, self._stop)
        logging.info("Local runtime supervisor listening on %s", self.socket_path)
        try:
            while not self._stopping.is_set():
                try:
                    connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stopping.is_set() or exc.errno in (
                        errno.EBADF,
                        errno.EINVAL,
                    ):
                        break
                    raise
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(connection,),
                    daemon=True,
                )
                with self._threads_lock:
                    self._threads.add(thread)
                thread.start()
        finally:
            logging.info("event=supervisor_stopping socket=%s", self.socket_path)
            self.manager.close()
            if self._listener is not None:
                try:
                    self._listener.close()
                except OSError:
                    pass
                self._listener = None
            self.socket_path.unlink(missing_ok=True)
            logging.info("event=supervisor_stopped")

    def close(self):
        self._stop()
