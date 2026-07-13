"""Command-line interface for the local MLX-LM runtime."""

import argparse
import json
import logging
import sys

from mlx_lm_runtime.client import UnixRuntimeClient
from mlx_lm_runtime.paths import DEFAULT_SOCKET_PATH
from mlx_lm_runtime.types import (
    GenerateRequest,
    GenerationDelta,
    GenerationFinished,
    GenerationStarted,
)

from .launchd import install_launch_agent, uninstall_launch_agent


def _parse_size(value):
    text = str(value).strip().upper()
    multipliers = {
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
    }
    for suffix in ("GB", "MB", "KB", "G", "M", "K"):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multipliers[suffix])
    return int(text)


def _client(args):
    return UnixRuntimeClient(args.socket, timeout=args.timeout)


def _serve(args):
    from .daemon import RuntimeDaemon
    from .engine import SequentialRuntime

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    engine = SequentialRuntime(
        args.model,
        adapter_path=args.adapter_path,
        trust_remote_code=args.trust_remote_code,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes,
        prefill_step_size=args.prefill_step_size,
    )
    RuntimeDaemon(engine, args.socket).serve_forever()


def _generate(args):
    messages = None
    if args.messages_json is not None:
        messages = json.loads(args.messages_json)
    request = GenerateRequest(
        prompt=args.prompt,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        stop=tuple(args.stop),
        seed=args.seed,
    )
    for event in _client(args).stream_events(request):
        if isinstance(event, GenerationStarted) and args.verbose:
            print(
                "[prompt={} cached={}]".format(
                    event.prompt_tokens, event.cached_tokens
                ),
                file=sys.stderr,
            )
        elif isinstance(event, GenerationDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, GenerationFinished):
            print()
            if args.verbose:
                print(json.dumps(event.__dict__, indent=2), file=sys.stderr)


def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mlx_lm.runtime",
        description="Run and access a sequential MLX-LM Unix-socket daemon.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the foreground daemon.")
    serve.add_argument("--model", required=True)
    serve.add_argument("--adapter-path")
    serve.add_argument("--socket", default=str(DEFAULT_SOCKET_PATH))
    serve.add_argument("--prompt-cache-size", type=int, default=4)
    serve.add_argument("--prompt-cache-bytes", type=_parse_size, default=2 * 1024**3)
    serve.add_argument("--prefill-step-size", type=int, default=2048)
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--verbose", action="store_true")
    serve.set_defaults(handler=_serve)

    def add_client_options(command):
        command.add_argument("--socket", default=str(DEFAULT_SOCKET_PATH))
        command.add_argument("--timeout", type=float)

    generate = subparsers.add_parser(
        "generate", help="Generate text through the daemon."
    )
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--messages-json")
    generate.add_argument("--max-tokens", type=int, default=256)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--min-p", type=float, default=0.0)
    generate.add_argument("--stop", action="append", default=[])
    generate.add_argument("--seed", type=int)
    generate.add_argument("--verbose", action="store_true")
    add_client_options(generate)
    generate.set_defaults(handler=_generate)

    for name in ("health", "status", "clear-cache", "shutdown"):
        command = subparsers.add_parser(name)
        add_client_options(command)
        method = name.replace("-", "_")
        command.set_defaults(
            handler=lambda args, method=method: _print_json(
                getattr(_client(args), method)()
            )
        )

    install = subparsers.add_parser(
        "install-launch-agent", help="Install and start the per-user LaunchAgent."
    )
    install.add_argument("--model", required=True)
    install.add_argument("--adapter-path")
    install.add_argument("--socket", default=str(DEFAULT_SOCKET_PATH))
    install.add_argument("--prompt-cache-size", type=int, default=4)
    install.add_argument("--prompt-cache-bytes", type=_parse_size, default=2 * 1024**3)
    install.add_argument("--prefill-step-size", type=int, default=2048)
    install.add_argument("--trust-remote-code", action="store_true")
    install.add_argument("--no-start", action="store_true")
    install.set_defaults(
        handler=lambda args: print(
            install_launch_agent(
                args.model,
                socket_path=args.socket,
                adapter_path=args.adapter_path,
                prompt_cache_size=args.prompt_cache_size,
                prompt_cache_bytes=args.prompt_cache_bytes,
                prefill_step_size=args.prefill_step_size,
                trust_remote_code=args.trust_remote_code,
                start=not args.no_start,
            )
        )
    )

    uninstall = subparsers.add_parser(
        "uninstall-launch-agent", help="Stop and remove the per-user LaunchAgent."
    )
    uninstall.set_defaults(handler=lambda _: uninstall_launch_agent())
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ConnectionError, OSError) as exc:
        client_commands = {
            "generate",
            "health",
            "status",
            "clear-cache",
            "shutdown",
        }
        description = (
            "Local runtime connection failed"
            if args.command in client_commands
            else "Local runtime command failed"
        )
        raise SystemExit("{}: {}".format(description, exc)) from exc


if __name__ == "__main__":
    main()
