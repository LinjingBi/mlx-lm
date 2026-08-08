"""Simplified ResponseGenerator twin for the local Unix runtime.

Owns one fixed model, admits concurrent ``GenerateRequest`` work into
``BatchGenerator``, and falls back to sequential ``stream_generate`` for
seeded or otherwise non-batchable requests. Transport stays outside this
module: callers supply request IDs and receive events through a sink.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx

from ..generate import (
    BatchGenerator,
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
from mlx_lm_runtime.types import GenerateRequest

EventSink = Callable[[str, str, Dict[str, Any]], None]


class _TimeBudget:
    """Wall-clock slice so the loop can admit new work between decode bursts."""

    def __init__(self, budget: float = 0.5):
        self._budget = budget
        self._start = None

    def __iter__(self):
        self._start = time.time()
        return self

    def __next__(self):
        if time.time() - self._start > self._budget:
            raise StopIteration()
        return None


@dataclass
class _ActiveRequest:
    request_id: str
    request: GenerateRequest
    prompt_tokens: List[int]
    cached_tokens: int
    started_at: float
    detokenizer: Any
    stop_words: Tuple[str, ...]
    text_filter: Optional[TextStateMachine] = None
    text_state: Any = None
    uid: Optional[int] = None
    first_token_at: Optional[float] = None
    generation_tokens: int = 0
    cancelled: bool = False
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0


@dataclass
class _Pending:
    request_id: str
    request: GenerateRequest


def _model_is_batchable(model) -> bool:
    try:
        return all(hasattr(c, "merge") for c in make_prompt_cache(model))
    except Exception:
        return False


class LocalResponseGenerator:
    """Admit, batch, and stream local-runtime generation events for one model."""

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
        prompt_concurrency: int = 8,
        decode_concurrency: int = 32,
        event_sink: Optional[EventSink] = None,
    ):
        if prompt_cache_size <= 0:
            raise ValueError("prompt_cache_size must be greater than zero.")
        if prompt_cache_bytes <= 0:
            raise ValueError("prompt_cache_bytes must be greater than zero.")
        if prefill_step_size <= 0:
            raise ValueError("prefill_step_size must be greater than zero.")
        if prompt_concurrency <= 0:
            raise ValueError("prompt_concurrency must be greater than zero.")
        if decode_concurrency <= 0:
            raise ValueError("decode_concurrency must be greater than zero.")

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
        self.prompt_concurrency = prompt_concurrency
        self.decode_concurrency = decode_concurrency
        self.prompt_cache_bytes = prompt_cache_bytes
        self.prompt_cache = LRUPromptCache(
            max_size=prompt_cache_size,
            max_bytes=prompt_cache_bytes,
        )
        self._is_batchable = _model_is_batchable(self.model)
        self._event_sink = event_sink or (lambda *_args, **_kwargs: None)

        self._requests: Queue = Queue()
        self._unprocessed: List[_Pending] = []
        self._active: Dict[str, _ActiveRequest] = {}
        self._uid_to_request_id: Dict[int, str] = {}
        self._active_lock = Lock()
        self._batch_generator: Optional[BatchGenerator] = None
        self._drain_batch = False
        self._stop = False
        self._time_budget = _TimeBudget()
        self._generation_stream = mx.default_stream(mx.default_device())
        self._thread = Thread(
            target=self._run, name="local-response-generator", daemon=True
        )
        self._thread.start()

    def set_event_sink(self, event_sink: EventSink):
        self._event_sink = event_sink

    def _emit(self, request_id: str, event_name: str, data: Dict[str, Any]):
        try:
            self._event_sink(request_id, event_name, data)
        except Exception:
            logging.exception(
                "Local response generator failed to emit %s for %s",
                event_name,
                request_id,
            )

    def _emit_error(self, request_id: str, message: str):
        self._emit(request_id, "error", {"message": message})

    def active_count(self) -> int:
        with self._active_lock:
            return len(self._active)

    def submit(self, request_id: str, request: GenerateRequest):
        request.validate()
        with self._active_lock:
            if len(self._active) >= self.decode_concurrency:
                raise RuntimeBusy(
                    "The local runtime is at decode concurrency cap ({}).".format(
                        self.decode_concurrency
                    )
                )
            # Reserve the slot before enqueue so concurrent submitters see it.
            self._active[request_id] = _ActiveRequest(
                request_id=request_id,
                request=request,
                prompt_tokens=[],
                cached_tokens=0,
                started_at=time.perf_counter(),
                detokenizer=None,
                stop_words=tuple(request.stop),
            )
        self._requests.put(_Pending(request_id, request))

    def cancel(self, request_id: str):
        with self._active_lock:
            active = self._active.get(request_id)
            if active is not None:
                active.cancelled = True

    def clear_cache(self):
        if self.active_count() > 0:
            raise RuntimeBusy("Cannot clear the cache during generation.")
        self.prompt_cache.trim_to(n_sequences=0, n_bytes=0)

    def status(self):
        with self._active_lock:
            active = len(self._active)
        return {
            "model": self.model_path,
            "adapter": self.adapter_path,
            "active": active > 0,
            "active_requests": active,
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "batchable": self._is_batchable,
            "prompt_cache_entries": len(self.prompt_cache),
            "prompt_cache_bytes": self.prompt_cache.nbytes,
            "prompt_cache_by_type": self.prompt_cache.stats_by_type(),
            "prefill_step_size": self.prefill_step_size,
        }

    def stop(self):
        self._stop = True
        self._thread.join(timeout=5)
        if self._batch_generator is not None:
            self._batch_generator.close()
            self._batch_generator = None

    def _tokenize(self, request: GenerateRequest) -> List[int]:
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

        if not uncached:
            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                return cache, prompt_tokens[-1:], len(prompt_tokens) - 1
            return make_prompt_cache(self.model), prompt_tokens, 0

        return cache, uncached, len(prompt_tokens) - len(uncached)

    def _is_request_batchable(self, request: GenerateRequest) -> bool:
        return self._is_batchable and request.seed is None

    def _next_pending(self, timeout=None) -> Optional[_Pending]:
        if self._unprocessed:
            return self._unprocessed.pop()
        try:
            if timeout is None:
                return self._requests.get_nowait()
            return self._requests.get(timeout=timeout)
        except Empty:
            return None

    def _drop_active(self, request_id: str):
        with self._active_lock:
            active = self._active.pop(request_id, None)
            if active is not None and active.uid is not None:
                self._uid_to_request_id.pop(active.uid, None)
            return active

    def _finish_request(
        self,
        active: _ActiveRequest,
        finish_reason: str,
        *,
        commit_cache=None,
        cache_key=None,
    ):
        if active.cancelled:
            self._drop_active(active.request_id)
            return
        if commit_cache is not None and cache_key is not None:
            self.prompt_cache.insert_cache(self.model_key, cache_key, commit_cache)
        completed_at = time.perf_counter()
        self._emit(
            active.request_id,
            "GenerationFinished",
            {
                "finish_reason": finish_reason,
                "prompt_tokens": len(active.prompt_tokens),
                "cached_tokens": active.cached_tokens,
                "generation_tokens": active.generation_tokens,
                "prompt_tps": active.prompt_tps,
                "generation_tps": active.generation_tps,
                "peak_memory_gb": active.peak_memory_gb or (mx.get_peak_memory() / 1e9),
                "ttft_seconds": (active.first_token_at or completed_at)
                - active.started_at,
                "total_seconds": completed_at - active.started_at,
            },
        )
        self._drop_active(active.request_id)

    def _ensure_batch_generator(self):
        if self._batch_generator is None:
            self._batch_generator = BatchGenerator(
                self.model,
                completion_batch_size=self.decode_concurrency,
                prefill_batch_size=self.prompt_concurrency,
                prefill_step_size=self.prefill_step_size,
                stream=self._generation_stream,
            )

    def _close_batch_generator(self):
        if self._batch_generator is not None:
            self._batch_generator.close()
            self._batch_generator = None
        self._drain_batch = False

    def _admit_batchable(self, pending: _Pending):
        with self._active_lock:
            active = self._active.get(pending.request_id)
        if active is None or active.cancelled:
            self._drop_active(pending.request_id)
            return

        try:
            prompt_tokens = self._tokenize(pending.request)
            if not prompt_tokens:
                raise ValueError("The rendered prompt contains no tokens.")
            cache, uncached, cached_tokens = self._prepare_cache(prompt_tokens)
        except Exception as exc:
            self._emit_error(pending.request_id, str(exc))
            self._drop_active(pending.request_id)
            return

        active.prompt_tokens = prompt_tokens
        active.cached_tokens = cached_tokens
        active.detokenizer = self.tokenizer.detokenizer
        if pending.request.stop:
            active.text_filter = TextStateMachine(
                {"normal": [(word, "normal") for word in pending.request.stop]}
            )
            active.text_state = active.text_filter.make_state("normal")

        self._emit(
            pending.request_id,
            "GenerationStarted",
            {
                "model": self.model_path,
                "prompt_tokens": len(prompt_tokens),
                "cached_tokens": cached_tokens,
            },
        )

        self._ensure_batch_generator()
        stop_matcher = make_stop_matcher(self.tokenizer, pending.request.stop)
        sampler = make_sampler(
            pending.request.temperature,
            top_p=pending.request.top_p,
            top_k=pending.request.top_k,
            min_p=pending.request.min_p,
        )
        (uid,) = self._batch_generator.insert(
            prompts=[uncached],
            max_tokens=[pending.request.max_tokens],
            caches=[cache],
            all_tokens=[prompt_tokens[:cached_tokens]],
            samplers=[sampler],
            stop_matchers=[stop_matcher],
        )
        active.uid = uid
        with self._active_lock:
            self._uid_to_request_id[uid] = pending.request_id

        active_bytes = self._batch_generator.prompt_cache_nbytes
        self.prompt_cache.trim_to(
            n_bytes=max(0, self.prompt_cache_bytes - active_bytes)
        )

    def _serve_sequential(self, pending: _Pending):
        with self._active_lock:
            active = self._active.get(pending.request_id)
        if active is None or active.cancelled:
            self._drop_active(pending.request_id)
            return

        generation = None
        try:
            prompt_tokens = self._tokenize(pending.request)
            if not prompt_tokens:
                raise ValueError("The rendered prompt contains no tokens.")
            cache, uncached_tokens, cached_tokens = self._prepare_cache(prompt_tokens)
            active.prompt_tokens = prompt_tokens
            active.cached_tokens = cached_tokens

            self._emit(
                pending.request_id,
                "GenerationStarted",
                {
                    "model": self.model_path,
                    "prompt_tokens": len(prompt_tokens),
                    "cached_tokens": cached_tokens,
                },
            )

            if pending.request.seed is not None:
                mx.random.seed(pending.request.seed)
            sampler = make_sampler(
                pending.request.temperature,
                top_p=pending.request.top_p,
                top_k=pending.request.top_k,
                min_p=pending.request.min_p,
            )
            stop_matcher = make_stop_matcher(self.tokenizer, pending.request.stop)
            stop_state = stop_matcher.make_state()
            text_filter = None
            text_state = None
            if pending.request.stop:
                text_filter = TextStateMachine(
                    {"normal": [(word, "normal") for word in pending.request.stop]}
                )
                text_state = text_filter.make_state("normal")

            cache_key = list(prompt_tokens)
            generation = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=uncached_tokens,
                max_tokens=pending.request.max_tokens,
                sampler=sampler,
                prompt_cache=cache,
                prefill_step_size=self.prefill_step_size,
            )

            finish_reason = None
            final_response = None
            for response in generation:
                if active.cancelled:
                    break
                if active.first_token_at is None:
                    active.first_token_at = time.perf_counter()
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
                active.generation_tokens = response.generation_tokens
                active.prompt_tps = response.prompt_tps
                active.generation_tps = response.generation_tps
                active.peak_memory_gb = response.peak_memory
                self._emit(
                    pending.request_id,
                    "GenerationDelta",
                    {
                        "text": text,
                        "token": response.token,
                        "finish_reason": finish_reason,
                    },
                )
                if finish_reason is not None:
                    break

            if active.cancelled:
                self._drop_active(pending.request_id)
                return
            if final_response is None or finish_reason is None:
                raise RuntimeError("Generation ended without a finish reason.")
            self._finish_request(
                active,
                finish_reason,
                commit_cache=cache,
                cache_key=cache_key,
            )
        except Exception as exc:
            self._emit_error(pending.request_id, str(exc))
            self._drop_active(pending.request_id)
        finally:
            if generation is not None:
                generation.close()

    def _step_batch(self):
        if self._batch_generator is None:
            return
        with self._active_lock:
            if not any(a.uid is not None for a in self._active.values()):
                if self._drain_batch:
                    self._close_batch_generator()
                return

        uids_to_remove = []
        for _ in self._time_budget:
            prompt_responses, gen_responses = self._batch_generator.next()
            if not prompt_responses and not gen_responses:
                break

            for response in gen_responses:
                request_id = self._uid_to_request_id.get(response.uid)
                if request_id is None:
                    continue
                with self._active_lock:
                    active = self._active.get(request_id)
                if active is None:
                    continue
                if active.cancelled:
                    uids_to_remove.append(response.uid)
                    continue

                if active.first_token_at is None:
                    active.first_token_at = time.perf_counter()
                active.generation_tokens += 1

                if response.finish_reason == "stop":
                    active.detokenizer.finalize()
                    text = active.detokenizer.last_segment
                elif response.finish_reason == "length":
                    active.detokenizer.add_token(response.token)
                    active.detokenizer.finalize()
                    text = active.detokenizer.last_segment
                else:
                    active.detokenizer.add_token(response.token)
                    text = active.detokenizer.last_segment

                if active.text_filter is not None:
                    active.text_state, text, _ = TextStateMachine.step(
                        active.text_state, text
                    )
                    if response.finish_reason == "stop":
                        active.text_state, _ = TextStateMachine.discard(
                            active.text_state
                        )
                    elif response.finish_reason is not None:
                        active.text_state, tail, _ = TextStateMachine.flush(
                            active.text_state
                        )
                        text += tail

                self._emit(
                    request_id,
                    "GenerationDelta",
                    {
                        "text": text,
                        "token": response.token,
                        "finish_reason": response.finish_reason,
                    },
                )

                if response.finish_reason is not None:
                    elapsed = time.perf_counter() - active.started_at
                    active.generation_tps = (
                        active.generation_tokens / elapsed if elapsed > 0 else 0.0
                    )
                    active.peak_memory_gb = mx.get_peak_memory() / 1e9
                    self._finish_request(
                        active,
                        response.finish_reason,
                        commit_cache=response.prompt_cache,
                        cache_key=response.all_tokens,
                    )

        if uids_to_remove:
            self._batch_generator.remove(uids_to_remove)
            for uid in uids_to_remove:
                request_id = self._uid_to_request_id.pop(uid, None)
                if request_id is not None:
                    self._drop_active(request_id)

        with self._active_lock:
            batch_empty = not any(a.uid is not None for a in self._active.values())
        if batch_empty and self._drain_batch:
            self._close_batch_generator()

    def _run(self):
        while not self._stop:
            pending = None
            if not self._drain_batch:
                with self._active_lock:
                    has_batch_work = any(
                        a.uid is not None for a in self._active.values()
                    )
                timeout = (
                    None
                    if (self._batch_generator is not None and has_batch_work)
                    else 0.1
                )
                pending = self._next_pending(timeout=timeout)

            if pending is not None:
                with self._active_lock:
                    active = self._active.get(pending.request_id)
                if active is None or active.cancelled:
                    self._drop_active(pending.request_id)
                    continue

                batchable = self._is_request_batchable(pending.request)
                if (
                    self._batch_generator is not None
                    and batchable
                    and not self._drain_batch
                ):
                    self._admit_batchable(pending)
                    continue

                if self._batch_generator is None:
                    if batchable:
                        self._ensure_batch_generator()
                        self._unprocessed.append(pending)
                    else:
                        self._serve_sequential(pending)
                    continue

                # Live batch cannot accept this request (seeded / non-batchable).
                self._drain_batch = True
                self._unprocessed.append(pending)
                continue

            if self._batch_generator is not None:
                self._step_batch()
            elif self._drain_batch:
                self._drain_batch = False
