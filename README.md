# MLX-LM Local Runtime

The local runtime keeps a lightweight per-user supervisor available on a Unix
domain socket. It starts a disposable MLX model worker on demand and unloads
that worker after an idle timeout, releasing model weights and prompt caches.
Short-lived Python programs and shell integrations call the supervisor instead
of hosting MLX themselves or using HTTP.

This interface is optimized for same-machine, latency-sensitive local work such
as shell-command completion. It supports **partial** continuous batching for
concurrent `GenerateRequest` calls against one resident model. It is not a
drop-in replacement for `mlx_lm.server` (see Scheduling limitations).

## Main features

- Direct inference through `BatchGenerator` / `stream_generate`; there is no
  HTTP, OpenAI-compatible request handling, or SSE layer.
- One model worker, loaded lazily and retained while it is active, with
  concurrent generates up to `--decode-concurrency`.
- Configurable idle eviction that leaves the public socket and control APIs
  available while model memory is released.
- Full-prompt requests with automatic nearest-prefix matching through
  `LRUPromptCache`.
- Transactional cache updates: a failed or disconnected request does not mutate
  an existing cached entry.
- Streaming and non-streaming Python APIs.
- A versioned, length-prefixed JSON protocol over an `AF_UNIX` stream socket.
- A foreground daemon suitable for supervision by macOS `launchd`.
- Health, lifecycle status, cache-clear, explicit unload, and shutdown
  operations.
- Socket directory mode `0700` and socket mode `0600` by default.

## Install this checkout

```sh
cd ~/mlx-lm-unix-runtime
git switch codex/unix-native-runtime

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

The examples below use `mlx-community/Qwen3-4B-Instruct-2507-4bit`, but a local
model directory can be supplied instead.

## Run the daemon in the foreground

```sh
mlx_lm.runtime serve \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt-cache-size 4 \
  --prompt-cache-bytes 2G \
  --prefill-step-size 2048 \
  --idle-timeout 300
```

The supervisor does not load the model until the first generation request.
After five idle minutes, the default `--idle-timeout 300` terminates the model
worker and discards its in-memory prompt caches. The next generation reloads
the model. Set `--idle-timeout 0` to keep the worker resident indefinitely.

The default socket is:

```text
~/Library/Caches/mlx-lm-runtime/runtime.sock
```

Use `--socket` on both the daemon and client commands to override it.

## Python client

The client does not import MLX or load model weights.

```python
from mlx_lm_runtime import UnixRuntimeClient

client = UnixRuntimeClient()
result = client.generate(
    messages=[
        {"role": "system", "content": "Return one shell command."},
        {"role": "user", "content": "Complete: git che"},
    ],
    max_tokens=64,
    temperature=0,
    stop=("\n",),
)

print(result.text)
print("cached tokens:", result.started.cached_tokens)
print("TTFT:", result.finished.ttft_seconds)
```

Streaming returns token deltas:

```python
for delta in client.stream_generate(
    prompt="Complete this command: git che",
    max_tokens=64,
    temperature=0,
    stop=("\n",),
):
    print(delta.text, end="", flush=True)
```

Send the complete prompt or complete message history on every request. The
daemon owns tokenization and determines the reusable token prefix. Do not send
only a manually calculated suffix.

## Command-line client

```sh
mlx_lm.runtime health
mlx_lm.runtime status

mlx_lm.runtime generate \
  --prompt "Complete this command: git che" \
  --max-tokens 64 \
  --temperature 0 \
  --stop $'\n' \
  --verbose

mlx_lm.runtime clear-cache
mlx_lm.runtime unload
mlx_lm.runtime shutdown
```

Chat messages can be passed as JSON:

```sh
mlx_lm.runtime generate \
  --messages-json '[{"role":"user","content":"Say hello"}]' \
  --max-tokens 32
```

The model tokenizer must provide a chat template for message requests. Models
without one can still be used with a rendered raw prompt.

## Query runtime status

`health` reports whether the lightweight supervisor can accept requests. It
continues to return `status: ok` when the model worker is unloaded:

```sh
mlx_lm.runtime health
```

`status` reports model residency, worker activity, the idle deadline, cache
usage, and measured cold-start latency without loading or waking the model:

```sh
mlx_lm.runtime status
```

The single `worker_state` follows this lifecycle:

```text
unloaded -> loading -> ready -> busy -> ready -> unloading -> unloaded
                    \-> failed                 \-> failed
```

- `unloaded`: the model is not resident; the next request is cold.
- `loading`: a worker is loading the model for a waiting request.
- `ready`: the model is resident and waiting for a request.
- `busy`: generation is active.
- `unloading`: the worker is exiting and releasing memory.
- `failed`: worker startup or execution failed; a later request can retry.

Important status fields include:

```json
{
  "service_state": "running",
  "worker_state": "ready",
  "model_resident": true,
  "next_request_cold": false,
  "idle_timeout_seconds": 300.0,
  "idle_for_seconds": 42.3,
  "unload_in_seconds": 257.7,
  "estimated_cold_start_ms": 3182.0,
  "worker_pid": 41237,
  "prompt_cache_entries": 2,
  "prompt_cache_bytes": 123456789
}
```

After idle eviction, `worker_state` is `unloaded`, `model_resident` is false,
`next_request_cold` is true, and `unload_reason` is `idle_timeout`. During a
generation, `worker_state` is `busy` and `request_running_for_ms` indicates how
long it has been active. The legacy `active` field remains available and is
equivalent to `worker_state == "busy"`.

Health and status requests are handled concurrently by the supervisor, so they
remain responsive during model loading and generation. Multiple generate
connections can make progress together up to `--decode-concurrency`; beyond
that cap a new generate receives a `busy` error.

## Install as a macOS LaunchAgent

Run this command from the Python environment where the checkout is installed:

```sh
mlx_lm.runtime install-launch-agent \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt-cache-size 4 \
  --prompt-cache-bytes 2G \
  --idle-timeout 300
```

It writes and starts:

```text
~/Library/LaunchAgents/com.mlx-lm.local-runtime.plist
```

Logs are written under:

```text
~/Library/Logs/mlx-lm-runtime/
```

Structured lifecycle and timing events are written to `runtime.log`, which is
rotated at 10 MiB with five backups. `stdout.log` and `stderr.log` remain
launchd-level fallback logs.

The LaunchAgent uses the absolute path of the Python interpreter that executes
the install command. Moving or deleting that environment will break the agent.

Remove it with:

```sh
mlx_lm.runtime uninstall-launch-agent
```

## Cache behavior

The runtime applies the model's chat template, tokenizes the complete request,
and looks up the nearest cached sequence. A shorter cached sequence is reused
directly. A longer cached branch may be copied and trimmed to its common prefix
when the model's cache type supports trimming.

Only the uncached suffix is processed by the model. At successful completion,
the daemon inserts the prompt plus generated token IDs into the LRU. The cache
has both entry-count and byte limits.

Prompt caches are in memory only and are lost when the worker is explicitly
unloaded, reaches its idle timeout, fails, or the supervisor exits. Cache reuse
also depends on stable prompt serialization: changing a token near the
beginning of a prompt invalidates everything after it.

Some hybrid, recurrent, sliding-window, or model-specific cache implementations
cannot be trimmed safely. Such models may reuse fewer prefixes or fall back to
full prompt processing. The local runtime inherits those constraints from
`mlx-lm`; the Unix transport does not change them.

## Protocol

Each socket message contains a four-byte unsigned big-endian length followed by
a UTF-8 JSON object. Protocol version 1 supports these operations:

- `health`
- `status`
- `generate`
- `clear_cache`
- `shutdown`

Generation emits `started`, `delta`, and `finished` frames. Every request and
response contains the protocol version and request ID. Frames are limited to 8
MiB by default. Python `pickle` is intentionally not used.

One connection carries one operation and closes after the response. This keeps
the client simple and works well for short-lived shell processes.

## Gaps compared with `mlx_lm.server`

| Capability | Local runtime | `mlx_lm.server` |
|---|---|---|
| Active inference | Up to decode concurrency | Multiple requests |
| Continuous batching | **Partial** — same-model `GenerateRequest` only | Uses `BatchGenerator` |
| Prompt/decode concurrency | Configurable | Configurable |
| Scheduling | Local twin of ResponseGenerator | Batched request scheduler |
| Transport | Same-machine Unix socket | OpenAI-like HTTP API |
| OpenAI SDK compatibility | No | Yes |
| Network or browser access | No | HTTP, streaming, and CORS |
| Resident models | One fixed startup model | Configured/on-demand model loading |
| Per-request model or adapter | No | Supported |
| Prefix LRU prompt cache | Yes | Yes |
| Batched segment checkpoints | No | System/user/assistant segments |
| Streaming text | Yes | Yes |
| Tool-call response formatting | No | Supported |
| Reasoning-content separation | No | Supported |
| Logprobs and top-logprobs | No | Supported |
| Logit bias and penalties | No | Supported |
| Draft/speculative generation | No | Supported on the sequential path |
| Distributed inference | No | Supported |
| Status during generation | Supervisor remains responsive | HTTP worker remains responsive |
| Cache persistence after exit | No | No by default |

The local runtime currently exposes temperature, top-p, top-k, min-p, seed,
maximum tokens, and stop strings. The narrower surface keeps the generate API
small and auditable.

## Scheduling limitations

Continuous batching here is intentionally narrower than `mlx_lm.server`:

| Area | Local runtime | `mlx_lm.server` |
|---|---|---|
| Continuous batching | Concurrent `generate` ops on the resident model | Yes |
| Models in one process | One fixed startup model/adapter | Multi-model / on-demand load; drain on model switch |
| Request API | `GenerateRequest` only | Full OpenAI-ish args (tools, logprobs, penalties, draft, …) |
| Seeded requests | Sequential / drain — not batched | Same |
| Draft / speculative | Not supported | Sequential path |
| Segmented prompt-cache checkpoints | Not in first pass (flat cache) | System/user/assistant segments |
| Transport | Unix, one op per connection; concurrency = many connections | HTTP + SSE |
| Backpressure | `busy` at decode-concurrency cap | Queue into `ResponseGenerator` (HTTP threads wait) |
| Distributed | No | Optional |

## When to use which interface

Use the local runtime when:

- all callers are on the same Mac;
- client processes are short lived;
- one resident model is enough;
- the narrow `GenerateRequest` surface is enough;
- partial continuous batching (same model, capped concurrency) is acceptable.

Use `mlx_lm.server` when multi-model serving, OpenAI API compatibility, rich
sampling/tool features, or unconstrained concurrent HTTP clients are required.

## Maintaining the downstream patch

The fork keeps `main` as a clean mirror of `ml-explore/mlx-lm:main`. Runtime
changes live only on `codex/unix-native-runtime`.

Two GitHub Actions workflows maintain this arrangement:

- `Local Runtime CI` runs formatting, lint, the focused runtime tests, a
  lightweight-client import check, and a wheel-content check on patched pushes
  and pull requests.
- `Sync Upstream Into Local Runtime` runs every Monday at 03:17 Asia/Shanghai
  and on manual dispatch. It fast-forwards fork `main`, creates an
  `automation/sync-<upstream-sha>` branch, merges upstream into that candidate,
  validates it, and opens a pull request into the patched branch.

The sync workflow does not rebase, force-push, or modify the patched branch
directly. Merge conflicts and test failures stop the run for manual attention.
Because scheduled GitHub Actions run from the repository's default branch, the
fork should use `codex/unix-native-runtime` as its default branch.
