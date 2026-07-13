import os
import tempfile
import threading
import time
import unittest

from mlx_lm_runtime.client import RuntimeRemoteError, UnixRuntimeClient
from mlx_lm_runtime.types import (
    GenerateRequest,
    GenerationDelta,
    GenerationFinished,
    GenerationStarted,
)

from mlx_lm.local_runtime.daemon import RuntimeDaemon


class FakeEngine:
    def __init__(self):
        self.cleared = False
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        yield GenerationStarted("fake-model", 3, 2)
        yield GenerationDelta("done", 9, "length")
        yield GenerationFinished("length", 3, 2, 1, 100.0, 50.0, 0.1, 0.01, 0.03)

    def status(self):
        return {"model": "fake-model", "active": False}

    def clear_cache(self):
        self.cleared = True


class TestLocalRuntimeDaemon(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.chmod(self.temp_dir.name, 0o755)
        self.socket_path = os.path.join(self.temp_dir.name, "runtime.sock")
        self.engine = FakeEngine()
        self.daemon = RuntimeDaemon(self.engine, self.socket_path)
        self.thread = threading.Thread(
            target=self.daemon.serve_forever,
            kwargs={"install_signal_handlers": False},
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + 2
        while not os.path.exists(self.socket_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.client = UnixRuntimeClient(self.socket_path, timeout=2)

    def tearDown(self):
        if self.thread.is_alive():
            try:
                self.client.shutdown()
            except OSError:
                self.daemon.close()
        self.thread.join(timeout=2)
        self.temp_dir.cleanup()

    def test_control_operations_and_permissions(self):
        self.assertEqual(self.client.health()["status"], "ok")
        self.assertEqual(self.client.status()["model"], "fake-model")
        self.assertTrue(self.client.clear_cache()["cache_cleared"])
        self.assertTrue(self.engine.cleared)
        self.assertEqual(os.stat(self.temp_dir.name).st_mode & 0o777, 0o755)
        self.assertEqual(os.stat(self.socket_path).st_mode & 0o777, 0o600)

    def test_generate_and_stream(self):
        request = GenerateRequest(prompt="abc", max_tokens=4)
        result = self.client.generate(request)
        self.assertEqual(result.text, "done")
        self.assertEqual(result.started.cached_tokens, 2)
        self.assertEqual(result.finished.finish_reason, "length")
        self.assertEqual(self.engine.requests, [request])

        deltas = list(self.client.stream_generate(prompt="abc", max_tokens=4))
        self.assertEqual([delta.text for delta in deltas], ["done"])

    def test_invalid_operation_returns_error(self):
        with self.assertLogs(level="ERROR"):
            with self.assertRaises(RuntimeRemoteError) as context:
                self.client._request("not_an_operation")
        self.assertEqual(context.exception.code, "invalid_request")


if __name__ == "__main__":
    unittest.main()
