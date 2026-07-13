# MLX-LM Local Runtime

The local runtime keeps one MLX model and its prompt caches resident in a
per-user macOS daemon. Short-lived Python programs and shell integrations call
the daemon through a Unix domain socket instead of HTTP.

This interface is deliberately optimized for sequential, latency-sensitive
local work such as shell-command completion. It is not a replacement for
`mlx_lm.server` when several requests need to run concurrently.

## Main features

- Direct inference through `mlx_lm.stream_generate`; there is no HTTP,
  OpenAI-compatible request handling, or SSE layer.
- One model loaded once and retained for the lifetime of the daemon.
- Full-prompt requests with automatic nearest-prefix matching through
  `LRUPromptCache`.
- Transactional cache updates: a failed or disconnected request does not mutate
  an existing cached entry.
- Streaming and non-streaming Python APIs.
- A versioned, length-prefixed JSON protocol over an `AF_UNIX` stream socket.
- A foreground daemon suitable for supervision by macOS `launchd`.
- Health, status, cache-clear, and shutdown operations.
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
  --prefill-step-size 2048
```

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

## Install as a macOS LaunchAgent

Run this command from the Python environment where the checkout is installed:

```sh
mlx_lm.runtime install-launch-agent \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt-cache-size 4 \
  --prompt-cache-bytes 2G
```

It writes and starts:

```text
~/Library/LaunchAgents/com.mlx-lm.local-runtime.plist
```

Logs are written under:

```text
~/Library/Logs/mlx-lm-runtime/
```

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

Prompt caches are in memory only and are lost when the daemon exits. Cache reuse
also depends on stable prompt serialization: changing a token near the beginning
of a prompt invalidates everything after it.

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
| Active inference | Exactly one | Multiple requests |
| Continuous batching | Not supported | Uses `BatchGenerator` |
| Prompt/decode concurrency | Not supported | Configurable |
| Scheduling | Connections wait serially | Batched request scheduler |
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
| Status during generation | Waits behind generation | HTTP worker remains responsive |
| Cache persistence after exit | No | No by default |

The daemon listens with a socket backlog of one but does not execute requests in
parallel. A second client may connect and then wait until the current generation
finishes. This is intentional: one worker owns all MLX execution and all mutable
cache state.

The local runtime currently exposes temperature, top-p, top-k, min-p, seed,
maximum tokens, and stop strings. The narrower surface keeps the sequential path
small and auditable.

## When to use which interface

Use the local runtime when:

- all callers are on the same Mac;
- client processes are short lived;
- latency matters more than aggregate throughput;
- inference is naturally sequential;
- one resident model is enough.

Use `mlx_lm.server` when several callers must make progress simultaneously,
continuous batching matters, OpenAI API compatibility is required, or the
caller is not able to access a local Unix socket.

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
