import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from mlx_lm.local_runtime.errors import RuntimeBusy
from mlx_lm_runtime.types import GenerateRequest


class FakeTokenizer:
    has_chat_template = True
    eos_token_ids = {0}

    def encode(self, text, add_special_tokens=True):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    @property
    def detokenizer(self):
        detok = MagicMock()
        detok.last_segment = "x"
        return detok


class FakePromptCache:
    def __init__(self):
        self.insertions = []
        self.nbytes = 0

    def fetch_nearest_cache(self, model, tokens):
        return None, list(tokens)

    def insert_cache(self, model, tokens, cache, cache_type="assistant"):
        self.insertions.append((model, list(tokens), cache))

    def trim_to(self, **kwargs):
        pass

    def stats_by_type(self):
        return {}

    def __len__(self):
        return len(self.insertions)


def _response(text, token, finish_reason, count):
    return types.SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish_reason,
        generation_tokens=count,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=0.1,
    )


class TestLocalResponseGenerator(unittest.TestCase):
    def _patches(self, batchable=False):
        return (
            patch(
                "mlx_lm.local_runtime.response_generator.load",
                return_value=(object(), FakeTokenizer()),
            ),
            patch(
                "mlx_lm.local_runtime.response_generator._model_is_batchable",
                return_value=batchable,
            ),
            patch(
                "mlx_lm.local_runtime.response_generator.make_prompt_cache",
                return_value=[object()],
            ),
        )

    def _make_generator(self, events, **kwargs):
        from mlx_lm.local_runtime.response_generator import LocalResponseGenerator

        def sink(request_id, event_name, data):
            events.append((request_id, event_name, data))

        with (
            self._patches(batchable=False)[0],
            self._patches(batchable=False)[1],
            self._patches(batchable=False)[2],
        ):
            generator = LocalResponseGenerator(
                "fake-model",
                prompt_cache_bytes=1024,
                decode_concurrency=kwargs.get("decode_concurrency", 2),
                prompt_concurrency=kwargs.get("prompt_concurrency", 2),
                event_sink=sink,
            )
        generator.prompt_cache = FakePromptCache()
        generator._is_batchable = False
        return generator

    def test_sequential_path_emits_started_delta_finished(self):
        events = []
        generator = self._make_generator(events)

        def fake_stream_generate(**kwargs):
            yield _response("x", 4, "length", 1)

        try:
            with (
                patch(
                    "mlx_lm.local_runtime.response_generator.stream_generate",
                    side_effect=fake_stream_generate,
                ),
                patch(
                    "mlx_lm.local_runtime.response_generator.make_prompt_cache",
                    return_value=[object()],
                ),
            ):
                generator.submit("r1", GenerateRequest(prompt="abc", max_tokens=1))
                deadline = time.monotonic() + 2
                while (
                    not any(e[1] == "GenerationFinished" for e in events)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
        finally:
            generator.stop()

        names = [event[1] for event in events if event[0] == "r1"]
        self.assertEqual(
            names,
            ["GenerationStarted", "GenerationDelta", "GenerationFinished"],
        )
        self.assertEqual(generator.prompt_cache.insertions[0][1], [97, 98, 99, 4])

    def test_cancel_skips_cache_commit(self):
        events = []
        generator = self._make_generator(events)
        started = threading.Event()

        def fake_stream_generate(**kwargs):
            started.wait(timeout=2)
            yield _response("x", 4, None, 1)
            yield _response("y", 5, "length", 2)

        try:
            with (
                patch(
                    "mlx_lm.local_runtime.response_generator.stream_generate",
                    side_effect=fake_stream_generate,
                ),
                patch(
                    "mlx_lm.local_runtime.response_generator.make_prompt_cache",
                    return_value=[object()],
                ),
            ):
                generator.submit("r1", GenerateRequest(prompt="abc", max_tokens=8))
                deadline = time.monotonic() + 2
                while generator.active_count() == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                generator.cancel("r1")
                started.set()
                deadline = time.monotonic() + 2
                while generator.active_count() > 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
        finally:
            generator.stop()

        self.assertEqual(generator.prompt_cache.insertions, [])
        self.assertFalse(any(event[1] == "GenerationFinished" for event in events))

    def test_decode_concurrency_cap(self):
        events = []
        generator = self._make_generator(events, decode_concurrency=1)
        release = threading.Event()

        def fake_stream_generate(**kwargs):
            release.wait(timeout=2)
            yield _response("x", 4, "length", 1)

        try:
            with (
                patch(
                    "mlx_lm.local_runtime.response_generator.stream_generate",
                    side_effect=fake_stream_generate,
                ),
                patch(
                    "mlx_lm.local_runtime.response_generator.make_prompt_cache",
                    return_value=[object()],
                ),
            ):
                generator.submit("r1", GenerateRequest(prompt="abc", max_tokens=1))
                deadline = time.monotonic() + 2
                while generator.active_count() < 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                with self.assertRaises(RuntimeBusy):
                    generator.submit("r2", GenerateRequest(prompt="xyz", max_tokens=1))
                release.set()
                deadline = time.monotonic() + 2
                while generator.active_count() > 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
        finally:
            generator.stop()

    def test_seeded_request_uses_sequential_path(self):
        events = []
        generator = self._make_generator(events)
        generator._is_batchable = True
        seen = {}

        def fake_stream_generate(**kwargs):
            seen["called"] = True
            yield _response("x", 4, "length", 1)

        try:
            with (
                patch(
                    "mlx_lm.local_runtime.response_generator.stream_generate",
                    side_effect=fake_stream_generate,
                ),
                patch(
                    "mlx_lm.local_runtime.response_generator.make_prompt_cache",
                    return_value=[object()],
                ),
                patch(
                    "mlx_lm.local_runtime.response_generator.BatchGenerator"
                ) as batch_cls,
            ):
                generator.submit(
                    "r1", GenerateRequest(prompt="abc", max_tokens=1, seed=7)
                )
                deadline = time.monotonic() + 2
                while (
                    not any(e[1] == "GenerationFinished" for e in events)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
        finally:
            generator.stop()

        self.assertTrue(seen.get("called"))
        batch_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
