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
from pathlib import Path

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
    """Own one disposable worker and expose a thread-safe lifecycle snapshot."""

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
        self._context = process_context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._state_lock = threading.Lock()
        self._operation_lock = threading.Lock()
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
            if self._closed or self._state != "ready":
                return
            self._idle_timer = timer
        timer.start()

    def _idle_expired(self):
        if self._closed:
            return
        if not self._operation_lock.acquire(blocking=False):
            if not self._closed:
                self._schedule_idle_timer()
            return
        try:
            with self._state_lock:
                if self._closed or self._state != "ready":
                    return
            self._stop_worker_locked("idle_timeout")
        finally:
            self._operation_lock.release()

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
        # Daemon workers must not outlive the supervisor; otherwise a leaked
        # child blocks process shutdown via multiprocessing's infinite join.
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
        logging.info("event=model_loaded duration_ms=%s pid=%s", load_ms, process.pid)

    def _stop_process(self):
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
                connection.send({"operation": "stop"})
                if connection.poll(self.stop_timeout):
                    connection.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
        self._stop_process()
        with self._state_lock:
            self._state = "unloaded"
            self._unload_reason = reason
            self._unloaded_at = _now_iso()
            self._worker_status = {}
        logging.info("event=worker_stopped reason=%s", reason)

    def stream(self, request_data):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusy("The local runtime only supports one active request.")
        self._cancel_idle_timer()
        cold_start = False
        finished = None
        try:
            with self._state_lock:
                cold_start = self._state not in ("ready", "busy")
            self._start_worker_locked()
            with self._state_lock:
                self._state = "busy"
                self._request_started = time.monotonic()
            logging.info("event=generation_started cold_start=%s", cold_start)
            self._connection.send({"operation": "generate", "data": request_data})
            while True:
                message = self._connection.recv()
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
                    raise RuntimeError(message.get("message", "Worker request failed."))
                else:
                    raise RuntimeError("Unexpected worker response {!r}.".format(kind))
            now = time.monotonic()
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
            with self._state_lock:
                self._state = "ready"
                self._request_started = None
                self._last_activity_monotonic = now
                self._last_activity_at = _now_iso()
            logging.info(
                "event=generation_finished cold_start=%s total_seconds=%s",
                cold_start,
                finished_data.get("total_seconds"),
            )
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
            interrupted = False
            with self._state_lock:
                ready = self._state == "ready"
                if self._state == "busy":
                    self._state = "failed"
                    self._unload_reason = "worker_failure"
                    self._request_started = None
                    interrupted = True
            if interrupted:
                self._stop_process()
            self._operation_lock.release()
            if ready and not self._closed:
                self._schedule_idle_timer()

    def clear_cache(self):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusy("Cannot clear the cache during generation.")
        try:
            with self._state_lock:
                state = self._state
            if state == "unloaded":
                return
            if state != "ready":
                raise RuntimeBusy(
                    "Cannot clear the cache while worker is {}.".format(state)
                )
            self._connection.send({"operation": "clear_cache"})
            message = self._connection.recv()
            if message.get("kind") != "cleared":
                raise RuntimeError(
                    message.get("message", "Could not clear worker cache.")
                )
            with self._state_lock:
                self._worker_status = message.get("status", {})
        finally:
            self._operation_lock.release()

    def unload(self):
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeBusy("Cannot unload the model during generation.")
        try:
            self._stop_worker_locked("explicit_unload")
        finally:
            self._operation_lock.release()

    def close(self):
        self._closed = True
        self._cancel_idle_timer()
        self._operation_lock.acquire()
        try:
            self._stop_worker_locked("supervisor_shutdown")
        finally:
            self._operation_lock.release()

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
                "sequential": True,
                "model_ready": self._state in ("ready", "busy"),
            }

    def status(self):
        now = time.monotonic()
        with self._state_lock:
            state = self._state
            process = self._process
            worker_status = dict(self._worker_status)
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
                "active": state == "busy",
                "prefill_step_size": self.config.get("prefill_step_size", 2048),
                "prompt_cache_entries": 0,
                "prompt_cache_bytes": 0,
                "prompt_cache_by_type": {
                    cache_type: {"n_sequences": 0, "n_bytes": 0}
                    for cache_type in ("assistant", "user", "system")
                },
            }
        status.update(worker_status)
        status["active"] = state == "busy"
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
        try:
            payload = recv_frame(connection)
            if payload is None:
                return
            request_id, operation, data = validate_request_frame(payload)
            if operation == "generate":
                for message in self.manager.stream(data):
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
            self.manager.close()
            if self._listener is not None:
                try:
                    self._listener.close()
                except OSError:
                    pass
                self._listener = None
            self.socket_path.unlink(missing_ok=True)

    def close(self):
        self._stop()
