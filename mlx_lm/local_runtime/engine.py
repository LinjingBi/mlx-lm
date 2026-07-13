"""Single-model, single-request generation engine for the local runtime."""

import copy
import time
from threading import Lock
from typing import Iterator, Union

import mlx.core as mx

from ..generate import (
    StopSequenceMatcher,
    TextStateMachine,
    make_stop_matcher,
    stream_generate,
)
from ..models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from ..sample_utils import make_sampler
from ..utils import load
from .errors import RuntimeBusy
from mlx_lm_runtime.types import (
    GenerateRequest,
    GenerationDelta,
    GenerationFinished,
    GenerationStarted,
)


RuntimeEvent = Union[GenerationStarted, GenerationDelta, GenerationFinished]


class SequentialRuntime:
    """Keep one model resident and execute generations strictly in sequence.

    The caller always supplies a complete raw prompt or chat history. The runtime
    tokenizes it, finds the nearest cached prefix, and sends only the uncached
    suffix to ``mlx_lm.stream_generate``. A fetched cache is a private copy, so
    cache state is committed only after generation completes successfully.
    """

    def __init__(
        self,
        model_path: str,
        *,
        adapter_path=None,
        tokenizer_config=None,
        trust_remote_code: bool = False,
        prompt_cache_size: int = 4,
        prompt_cache_bytes: int = 2 * 1024 * 1024 * 1024,
        prefill_step_size: int = 2048,
    ):
        if prompt_cache_size <= 0:
            raise ValueError("prompt_cache_size must be greater than zero.")
        if prompt_cache_bytes <= 0:
            raise ValueError("prompt_cache_bytes must be greater than zero.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be greater than zero.")

        tokenizer_config = dict(tokenizer_config or {})
        tokenizer_config.setdefault("trust_remote_code", trust_remote_code)
        self.model, self.tokenizer = load(
            model_path,
            adapter_path=adapter_path,
            tokenizer_config=tokenizer_config,
            trust_remote_code=trust_remote_code,
        )
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.model_key = (self.model_path, self.adapter_path)
        self.prefill_step_size = prefill_step_size
        self.prompt_cache = LRUPromptCache(
            max_size=prompt_cache_size,
            max_bytes=prompt_cache_bytes,
        )
        self._generation_lock = Lock()

    def _tokenize(self, request: GenerateRequest):
        if request.prompt is not None:
            return list(self.tokenizer.encode(request.prompt))

        if not getattr(self.tokenizer, "has_chat_template", False):
            raise ValueError(
                "This tokenizer has no chat template; send a rendered raw prompt instead."
            )
        messages = copy.deepcopy(request.messages)
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **request.chat_template_kwargs,
            )
        )

    def _prepare_cache(self, prompt_tokens):
        cache, uncached = self.prompt_cache.fetch_nearest_cache(
            self.model_key, prompt_tokens
        )
        if cache is None:
            return make_prompt_cache(self.model), prompt_tokens, 0

        # Generation needs at least one input token. Exact prompt hits are rare
        # (completed assistant caches are normally longer than the next prompt),
        # but retaining the last token makes the behavior well-defined.
        if not uncached:
            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                return cache, prompt_tokens[-1:], len(prompt_tokens) - 1
            return make_prompt_cache(self.model), prompt_tokens, 0

        return cache, uncached, len(prompt_tokens) - len(uncached)

    def stream(self, request: GenerateRequest) -> Iterator[RuntimeEvent]:
        """Yield a start event, token deltas, and one finished event."""

        request.validate()
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeBusy("The local runtime only supports one active request.")

        generation = None
        request_started = time.perf_counter()
        try:
            prompt_tokens = self._tokenize(request)
            if not prompt_tokens:
                raise ValueError("The rendered prompt contains no tokens.")
            cache, uncached_tokens, cached_tokens = self._prepare_cache(prompt_tokens)

            yield GenerationStarted(
                model=self.model_path,
                prompt_tokens=len(prompt_tokens),
                cached_tokens=cached_tokens,
            )

            if request.seed is not None:
                mx.random.seed(request.seed)
            sampler = make_sampler(
                request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
            )
            stop_matcher = make_stop_matcher(self.tokenizer, request.stop)
            stop_state = stop_matcher.make_state()
            text_filter = None
            text_state = None
            if request.stop:
                text_filter = TextStateMachine(
                    {"normal": [(word, "normal") for word in request.stop]}
                )
                text_state = text_filter.make_state("normal")

            cache_key = list(prompt_tokens)
            generation = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=uncached_tokens,
                max_tokens=request.max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=self.prefill_step_size,
            )

            first_token_at = None
            final_response = None
            finish_reason = None
            for response in generation:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                final_response = response
                cache_key.append(response.token)
                stop_state, matched = StopSequenceMatcher.match(
                    stop_state, stop_matcher._trie, response.token
                )
                finish_reason = "stop" if matched else response.finish_reason

                text = response.text
                if text_filter is not None:
                    text_state, text, _ = TextStateMachine.step(text_state, text)
                    if finish_reason == "stop":
                        text_state, _ = TextStateMachine.discard(text_state)
                    elif finish_reason is not None:
                        text_state, tail, _ = TextStateMachine.flush(text_state)
                        text += tail

                yield GenerationDelta(
                    text=text,
                    token=response.token,
                    finish_reason=finish_reason,
                )
                if finish_reason is not None:
                    break

            if final_response is None or finish_reason is None:
                raise RuntimeError("Generation ended without a finish reason.")

            # Commit only after a complete generation. If the daemon stops
            # iterating because the client disconnects, closing this generator
            # jumps to ``finally`` without inserting the working cache copy.
            self.prompt_cache.insert_cache(self.model_key, cache_key, cache)
            completed_at = time.perf_counter()
            yield GenerationFinished(
                finish_reason=finish_reason,
                prompt_tokens=len(prompt_tokens),
                cached_tokens=cached_tokens,
                generation_tokens=final_response.generation_tokens,
                prompt_tps=final_response.prompt_tps,
                generation_tps=final_response.generation_tps,
                peak_memory_gb=final_response.peak_memory,
                ttft_seconds=(first_token_at or completed_at) - request_started,
                total_seconds=completed_at - request_started,
            )
        finally:
            if generation is not None:
                generation.close()
            self._generation_lock.release()

    def clear_cache(self):
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeBusy("Cannot clear the cache during generation.")
        try:
            self.prompt_cache.trim_to(n_sequences=0, n_bytes=0)
        finally:
            self._generation_lock.release()

    def status(self):
        return {
            "model": self.model_path,
            "adapter": self.adapter_path,
            "active": self._generation_lock.locked(),
            "prompt_cache_entries": len(self.prompt_cache),
            "prompt_cache_bytes": self.prompt_cache.nbytes,
            "prompt_cache_by_type": self.prompt_cache.stats_by_type(),
            "prefill_step_size": self.prefill_step_size,
        }
