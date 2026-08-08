import multiprocessing
import threading
import time
import unittest

from mlx_lm.local_runtime.supervisor import WorkerManager
from mlx_lm.local_runtime.errors import RuntimeBusy


def mux_fake_worker(connection, config):
    connection.send(
        {
            "kind": "ready",
            "load_ms": 1.0,
            "status": {
                "prompt_cache_entries": 0,
                "prompt_cache_bytes": 0,
                "prompt_cache_by_type": {},
            },
        }
    )
    cancelled = set()

    def run_one(request_id, delay):
        time.sleep(delay)
        if request_id in cancelled:
            return
        connection.send(
            {
                "kind": "event",
                "id": request_id,
                "event": "GenerationStarted",
                "data": {
                    "model": config["model"],
                    "prompt_tokens": 1,
                    "cached_tokens": 0,
                },
            }
        )
        connection.send(
            {
                "kind": "event",
                "id": request_id,
                "event": "GenerationDelta",
                "data": {
                    "text": request_id[:4],
                    "token": 1,
                    "finish_reason": "length",
                },
            }
        )
        connection.send(
            {
                "kind": "event",
                "id": request_id,
                "event": "GenerationFinished",
                "data": {
                    "finish_reason": "length",
                    "prompt_tokens": 1,
                    "cached_tokens": 0,
                    "generation_tokens": 1,
                    "prompt_tps": 1.0,
                    "generation_tps": 1.0,
                    "peak_memory_gb": 0.0,
                    "ttft_seconds": 0.01,
                    "total_seconds": 0.02,
                },
            }
        )
        connection.send(
            {
                "kind": "complete",
                "id": request_id,
                "status": {
                    "prompt_cache_entries": 1,
                    "prompt_cache_bytes": 8,
                    "prompt_cache_by_type": {},
                },
            }
        )

    while True:
        command = connection.recv()
        operation = command["operation"]
        if operation == "generate":
            threading.Thread(
                target=run_one,
                args=(command["id"], config.get("generation_delay", 0.05)),
                daemon=True,
            ).start()
        elif operation == "cancel":
            cancelled.add(command["id"])
        elif operation == "clear_cache":
            connection.send(
                {
                    "kind": "cleared",
                    "status": {
                        "prompt_cache_entries": 0,
                        "prompt_cache_bytes": 0,
                        "prompt_cache_by_type": {},
                    },
                }
            )
        elif operation == "stop":
            connection.send({"kind": "stopping", "status": {}})
            return


class TestWorkerMultiplexing(unittest.TestCase):
    def test_two_overlapping_streams_demux_by_request_id(self):
        manager = WorkerManager(
            {
                "model": "fake-model",
                "generation_delay": 0.1,
                "decode_concurrency": 4,
            },
            idle_timeout=0,
            stop_timeout=0.5,
            process_context=multiprocessing.get_context("spawn"),
            worker_target=mux_fake_worker,
        )
        results = [[], []]

        def run(index, prompt):
            results[index].extend(manager.stream({"prompt": prompt}))

        try:
            threads = [
                threading.Thread(target=run, args=(0, "one")),
                threading.Thread(target=run, args=(1, "two")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
            for events in results:
                self.assertEqual(events[0]["event"], "GenerationStarted")
                self.assertEqual(events[-1]["event"], "GenerationFinished")
                texts = [
                    event["data"]["text"]
                    for event in events
                    if event["event"] == "GenerationDelta"
                ]
                self.assertEqual(len(texts), 1)
        finally:
            manager.close()

    def test_clear_cache_blocked_while_active(self):
        manager = WorkerManager(
            {
                "model": "fake-model",
                "generation_delay": 0.3,
                "decode_concurrency": 4,
            },
            idle_timeout=0,
            stop_timeout=0.5,
            process_context=multiprocessing.get_context("spawn"),
            worker_target=mux_fake_worker,
        )
        thread = threading.Thread(
            target=lambda: list(manager.stream({"prompt": "hello"}))
        )
        try:
            thread.start()
            deadline = time.monotonic() + 2
            while (
                manager.status().get("active_requests", 0) < 1
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            with self.assertRaises(RuntimeBusy):
                manager.clear_cache()
            thread.join(timeout=2)
            manager.clear_cache()
        finally:
            manager.close()

    def test_cancel_on_stream_close(self):
        manager = WorkerManager(
            {
                "model": "fake-model",
                "generation_delay": 0.5,
                "decode_concurrency": 4,
            },
            idle_timeout=0,
            stop_timeout=0.5,
            process_context=multiprocessing.get_context("spawn"),
            worker_target=mux_fake_worker,
        )
        try:
            stream = manager.stream({"prompt": "hello"})
            next(stream)
            stream.close()
            deadline = time.monotonic() + 2
            while (
                manager.status().get("active_requests", 0) != 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(manager.status()["active_requests"], 0)
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
