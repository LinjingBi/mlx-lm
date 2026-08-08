import types
import unittest
from unittest.mock import patch


class FakeTokenizer:
    has_chat_template = True
    eos_token_ids = {0}

    def encode(self, text, add_special_tokens=True):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class FakePromptCache:
    def __init__(self, cache=None, suffix=None):
        self.cache = cache
        self.suffix = suffix
        self.insertions = []
        self.nbytes = 0

    def fetch_nearest_cache(self, model, tokens):
        return self.cache, list(tokens if self.suffix is None else self.suffix)

    def insert_cache(self, model, tokens, cache, cache_type="assistant"):
        self.insertions.append((model, tokens, cache))

    def trim_to(self, **kwargs):
        pass

    def stats_by_type(self):
        return {}

    def __len__(self):
        return len(self.insertions)


def response(text, token, finish_reason, count):
    return types.SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish_reason,
        generation_tokens=count,
        prompt_tps=100.0,
        generation_tps=50.0,
        peak_memory=0.1,
    )


class TestLocalRuntimeEngine(unittest.TestCase):
    def _runtime(self, prompt_cache):
        from mlx_lm.local_runtime.engine import SequentialRuntime

        with patch(
            "mlx_lm.local_runtime.engine.load",
            return_value=(object(), FakeTokenizer()),
        ):
            runtime = SequentialRuntime("fake-model", prompt_cache_bytes=1024)
        runtime.prompt_cache = prompt_cache
        return runtime

    def test_generation_commits_complete_cache_key(self):
        from mlx_lm_runtime.types import GenerateRequest, GenerationStarted

        cache = FakePromptCache()
        runtime = self._runtime(cache)
        observed = {}

        def fake_stream_generate(**kwargs):
            observed.update(kwargs)
            yield response("x", 4, "length", 1)

        with (
            patch(
                "mlx_lm.local_runtime.engine.make_prompt_cache", return_value=[object()]
            ),
            patch(
                "mlx_lm.local_runtime.engine.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            events = list(
                runtime.stream(
                    GenerateRequest(messages=[{"role": "user", "content": "hi"}])
                )
            )

        self.assertIsInstance(events[0], GenerationStarted)
        self.assertEqual(events[0].cached_tokens, 0)
        self.assertEqual(observed["prompt"], [1, 2, 3])
        self.assertEqual(cache.insertions[0][1], [1, 2, 3, 4])

    def test_cached_suffix_is_the_only_prompt_processed(self):
        from mlx_lm_runtime.types import GenerateRequest

        working_cache = [object()]
        cache = FakePromptCache(working_cache, [3])
        runtime = self._runtime(cache)
        observed = {}

        def fake_stream_generate(**kwargs):
            observed.update(kwargs)
            yield response("x", 4, "length", 1)

        with patch(
            "mlx_lm.local_runtime.engine.stream_generate",
            side_effect=fake_stream_generate,
        ):
            events = list(runtime.stream(GenerateRequest(prompt="\x01\x02\x03")))

        self.assertEqual(events[0].cached_tokens, 2)
        self.assertEqual(observed["prompt"], [3])
        self.assertIs(observed["prompt_cache"], working_cache)

    def test_closing_stream_before_finish_does_not_commit(self):
        from mlx_lm_runtime.types import GenerateRequest

        cache = FakePromptCache()
        runtime = self._runtime(cache)

        def fake_stream_generate(**kwargs):
            yield response("x", 4, None, 1)
            yield response("y", 5, "length", 2)

        with (
            patch(
                "mlx_lm.local_runtime.engine.make_prompt_cache", return_value=[object()]
            ),
            patch(
                "mlx_lm.local_runtime.engine.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            events = runtime.stream(GenerateRequest(prompt="abc"))
            next(events)
            next(events)
            events.close()

        self.assertEqual(cache.insertions, [])
        self.assertFalse(runtime.status()["active"])

    def test_stop_sequence_is_not_emitted(self):
        from mlx_lm_runtime.types import GenerateRequest, GenerationDelta

        cache = FakePromptCache()
        runtime = self._runtime(cache)

        def fake_stream_generate(**kwargs):
            for count, character in enumerate("aSTOP", start=1):
                yield response(character, ord(character), None, count)

        with (
            patch(
                "mlx_lm.local_runtime.engine.make_prompt_cache",
                return_value=[object()],
            ),
            patch(
                "mlx_lm.local_runtime.engine.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            events = list(runtime.stream(GenerateRequest(prompt="abc", stop=("STOP",))))

        text = "".join(
            event.text for event in events if isinstance(event, GenerationDelta)
        )
        self.assertEqual(text, "a")
        self.assertEqual(events[-1].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
