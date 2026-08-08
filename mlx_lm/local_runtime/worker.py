"""Disposable MLX worker process for the local runtime supervisor."""

import logging
import threading
import time
import traceback

from .errors import RuntimeBusy


def run_worker(connection, config):
    """Load one model and multiplex generate/cancel commands by request id."""

    send_lock = threading.Lock()

    def send(message):
        with send_lock:
            connection.send(message)

    generator = None

    def event_sink(request_id, event_name, data):
        if event_name == "error":
            send(
                {
                    "kind": "error",
                    "id": request_id,
                    "message": data.get("message", "Generation failed."),
                }
            )
            status = generator.status() if generator is not None else {}
            send({"kind": "complete", "id": request_id, "status": status})
            return
        send(
            {
                "kind": "event",
                "id": request_id,
                "event": event_name,
                "data": data,
            }
        )
        if event_name == "GenerationFinished":
            status = generator.status() if generator is not None else {}
            send({"kind": "complete", "id": request_id, "status": status})

    try:
        load_started = time.perf_counter()
        from .response_generator import LocalResponseGenerator

        generator = LocalResponseGenerator(
            config["model"],
            adapter_path=config.get("adapter_path"),
            trust_remote_code=config.get("trust_remote_code", False),
            prompt_cache_size=config.get("prompt_cache_size", 4),
            prompt_cache_bytes=config.get("prompt_cache_bytes", 2 * 1024**3),
            prefill_step_size=config.get("prefill_step_size", 2048),
            prompt_concurrency=config.get("prompt_concurrency", 8),
            decode_concurrency=config.get("decode_concurrency", 32),
            event_sink=event_sink,
        )
        send(
            {
                "kind": "ready",
                "load_ms": round((time.perf_counter() - load_started) * 1000, 3),
                "status": generator.status(),
            }
        )
    except BaseException as exc:
        send(
            {
                "kind": "startup_error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        connection.close()
        return

    try:
        while True:
            if not connection.poll(0.05):
                continue
            command = connection.recv()
            operation = command.get("operation")
            request_id = command.get("id")
            if operation == "generate":
                try:
                    from mlx_lm_runtime.types import GenerateRequest

                    request = GenerateRequest.from_dict(command["data"])
                    generator.submit(request_id, request)
                except RuntimeBusy as exc:
                    send(
                        {
                            "kind": "error",
                            "id": request_id,
                            "message": str(exc),
                            "code": "busy",
                        }
                    )
                except BaseException as exc:
                    send(
                        {
                            "kind": "error",
                            "id": request_id,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    send(
                        {
                            "kind": "complete",
                            "id": request_id,
                            "status": generator.status(),
                        }
                    )
            elif operation == "cancel":
                if request_id:
                    generator.cancel(request_id)
            elif operation == "clear_cache":
                try:
                    generator.clear_cache()
                    send({"kind": "cleared", "status": generator.status()})
                except RuntimeBusy as exc:
                    send({"kind": "error", "message": str(exc), "code": "busy"})
                except BaseException as exc:
                    send({"kind": "error", "message": str(exc)})
            elif operation == "stop":
                generator.stop()
                send({"kind": "stopping", "status": generator.status()})
                return
            else:
                send(
                    {
                        "kind": "error",
                        "message": "Unknown worker operation {!r}.".format(operation),
                    }
                )
    except (EOFError, BrokenPipeError, OSError):
        logging.info("Worker pipe closed.")
    finally:
        try:
            generator.stop()
        except Exception:
            pass
        connection.close()
