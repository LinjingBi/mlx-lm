"""Disposable MLX worker process for the local runtime supervisor."""

import dataclasses
import time
import traceback


def run_worker(connection, config):
    """Load one model, process sequential commands, and exit on request."""

    try:
        load_started = time.perf_counter()
        from .engine import SequentialRuntime

        engine = SequentialRuntime(
            config["model"],
            adapter_path=config.get("adapter_path"),
            trust_remote_code=config.get("trust_remote_code", False),
            prompt_cache_size=config.get("prompt_cache_size", 4),
            prompt_cache_bytes=config.get("prompt_cache_bytes", 2 * 1024**3),
            prefill_step_size=config.get("prefill_step_size", 2048),
        )
        connection.send(
            {
                "kind": "ready",
                "load_ms": round((time.perf_counter() - load_started) * 1000, 3),
                "status": engine.status(),
            }
        )
    except BaseException as exc:
        connection.send(
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
            command = connection.recv()
            operation = command.get("operation")
            if operation == "generate":
                try:
                    from mlx_lm_runtime.types import GenerateRequest

                    request = GenerateRequest.from_dict(command["data"])
                    for event in engine.stream(request):
                        connection.send(
                            {
                                "kind": "event",
                                "event": type(event).__name__,
                                "data": dataclasses.asdict(event),
                            }
                        )
                    connection.send({"kind": "complete", "status": engine.status()})
                except BaseException as exc:
                    connection.send(
                        {
                            "kind": "error",
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
            elif operation == "clear_cache":
                try:
                    engine.clear_cache()
                    connection.send({"kind": "cleared", "status": engine.status()})
                except BaseException as exc:
                    connection.send({"kind": "error", "message": str(exc)})
            elif operation == "stop":
                connection.send({"kind": "stopping", "status": engine.status()})
                return
            else:
                connection.send(
                    {
                        "kind": "error",
                        "message": "Unknown worker operation {!r}.".format(operation),
                    }
                )
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        connection.close()
