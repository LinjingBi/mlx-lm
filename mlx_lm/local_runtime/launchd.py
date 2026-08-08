"""Install and remove the per-user macOS LaunchAgent."""

import os
import plistlib
import subprocess
import sys

from mlx_lm_runtime.paths import (
    DEFAULT_SOCKET_PATH,
    LAUNCH_AGENT_LABEL,
    LAUNCH_AGENT_PATH,
    LOG_DIR,
)


def _service_target():
    return "gui/{}".format(os.getuid())


def install_launch_agent(
    model,
    *,
    socket_path=DEFAULT_SOCKET_PATH,
    adapter_path=None,
    prompt_cache_size=4,
    prompt_cache_bytes=2 * 1024 * 1024 * 1024,
    prefill_step_size=2048,
    idle_timeout=300,
    trust_remote_code=False,
    start=True,
):
    LOG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    LAUNCH_AGENT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    arguments = [
        sys.executable,
        "-m",
        "mlx_lm.local_runtime.cli",
        "serve",
        "--model",
        str(model),
        "--socket",
        str(socket_path),
        "--prompt-cache-size",
        str(prompt_cache_size),
        "--prompt-cache-bytes",
        str(prompt_cache_bytes),
        "--prefill-step-size",
        str(prefill_step_size),
        "--idle-timeout",
        str(idle_timeout),
    ]
    if adapter_path:
        arguments.extend(["--adapter-path", str(adapter_path)])
    if trust_remote_code:
        arguments.append("--trust-remote-code")

    payload = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": arguments,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(LOG_DIR / "stdout.log"),
        "StandardErrorPath": str(LOG_DIR / "stderr.log"),
    }
    with LAUNCH_AGENT_PATH.open("wb") as stream:
        plistlib.dump(payload, stream, sort_keys=True)

    if start:
        subprocess.run(
            ["launchctl", "bootout", _service_target(), str(LAUNCH_AGENT_PATH)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["launchctl", "bootstrap", _service_target(), str(LAUNCH_AGENT_PATH)],
            check=True,
        )
    return LAUNCH_AGENT_PATH


def uninstall_launch_agent():
    subprocess.run(
        ["launchctl", "bootout", _service_target(), str(LAUNCH_AGENT_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    LAUNCH_AGENT_PATH.unlink(missing_ok=True)
