import multiprocessing
import os
import tempfile
import threading
import time
import unittest

from mlx_lm.local_runtime.supervisor import RuntimeSupervisor, WorkerManager
from mlx_lm.local_runtime.errors import RuntimeBusy
from mlx_lm_runtime.client import UnixRuntimeClient


def fake_worker(connection, config):
    connection.send(
        {
            "kind": "ready",
            "load_ms": 12.5,
            "status": {
                "prompt_cache_entries": 0,
                "prompt_cache_bytes": 0,
                "prompt_cache_by_type": {},
            },
        }
    )
    while True:
        command = connection.recv()
        if command["operation"] == "generate":
            time.sleep(config.get("generation_delay", 0))
            connection.send(
                {
                    "kind": "event",
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
                    "event": "GenerationDelta",
                    "data": {"text": "ok", "token": 1, "finish_reason": "length"},
                }
            )
            connection.send(
                {
                    "kind": "event",
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
                    "status": {
                        "prompt_cache_entries": 1,
                        "prompt_cache_bytes": 64,
                        "prompt_cache_by_type": {},
                    },
                }
            )
        elif command["operation"] == "clear_cache":
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
        elif command["operation"] == "stop":
            connection.send({"kind": "stopping", "status": {}})
            return


class TestWorkerManager(unittest.TestCase):
    def make_manager(self, **kwargs):
        return WorkerManager(
            {"model": "fake-model", **kwargs},
            idle_timeout=0.15,
            stop_timeout=0.5,
            process_context=multiprocessing.get_context("spawn"),
            worker_target=fake_worker,
        )

    def test_lazy_start_and_idle_eviction(self):
        manager = self.make_manager()
        try:
            initial = manager.status()
            self.assertEqual(initial["worker_state"], "unloaded")
            self.assertTrue(initial["next_request_cold"])

            events = list(manager.stream({"prompt": "hello"}))
            self.assertEqual(events[-1]["event"], "GenerationFinished")
            ready = manager.status()
            pid = ready["worker_pid"]
            self.assertEqual(ready["worker_state"], "ready")
            self.assertFalse(ready["next_request_cold"])
            self.assertEqual(ready["prompt_cache_bytes"], 64)

            deadline = time.monotonic() + 2
            while (
                manager.status()["worker_state"] != "unloaded"
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            unloaded = manager.status()
            self.assertEqual(unloaded["worker_state"], "unloaded")
            self.assertEqual(unloaded["unload_reason"], "idle_timeout")
            self.assertIsNone(unloaded["worker_pid"])
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

            list(manager.stream({"prompt": "again"}))
            self.assertNotEqual(manager.status()["worker_pid"], pid)
            self.assertEqual(manager.status()["cold_start_count"], 2)
        finally:
            manager.close()
            self.assertFalse(
                any(
                    child.name == "mlx-lm-runtime-worker" and child.is_alive()
                    for child in multiprocessing.active_children()
                )
            )

    def test_status_is_observable_while_busy(self):
        manager = self.make_manager(generation_delay=0.3)
        result = []
        thread = threading.Thread(
            target=lambda: result.extend(manager.stream({"prompt": "hello"}))
        )
        try:
            thread.start()
            deadline = time.monotonic() + 2
            while (
                manager.status()["worker_state"] != "busy"
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertEqual(manager.status()["worker_state"], "busy")
            self.assertTrue(manager.status()["active"])
            with self.assertRaises(RuntimeBusy):
                list(manager.stream({"prompt": "second"}))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        finally:
            manager.close()


class TestRuntimeSupervisor(unittest.TestCase):
    def test_health_and_status_remain_available_during_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "runtime.sock")
            manager = WorkerManager(
                {"model": "fake-model", "generation_delay": 0.3},
                idle_timeout=0,
                stop_timeout=0.5,
                process_context=multiprocessing.get_context("spawn"),
                worker_target=fake_worker,
            )
            supervisor = RuntimeSupervisor(manager, socket_path)
            thread = threading.Thread(
                target=supervisor.serve_forever,
                kwargs={"install_signal_handlers": False},
                daemon=True,
            )
            thread.start()
            deadline = time.monotonic() + 2
            while not os.path.exists(socket_path) and time.monotonic() < deadline:
                time.sleep(0.01)
            client = UnixRuntimeClient(socket_path, timeout=2)
            generation = threading.Thread(
                target=lambda: client.generate(prompt="hello", max_tokens=1)
            )
            try:
                generation.start()
                deadline = time.monotonic() + 2
                while (
                    client.status()["worker_state"] != "busy"
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertEqual(client.health()["status"], "ok")
                self.assertEqual(client.status()["worker_state"], "busy")
                generation.join(timeout=2)
                self.assertFalse(generation.is_alive())
            finally:
                try:
                    client.shutdown()
                except OSError:
                    supervisor.close()
                thread.join(timeout=2)
                manager.close()
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
