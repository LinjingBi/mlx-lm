"""Default filesystem locations for the per-user local runtime."""

from pathlib import Path


RUNTIME_DIR = Path.home() / "Library" / "Caches" / "mlx-lm-runtime"
DEFAULT_SOCKET_PATH = RUNTIME_DIR / "runtime.sock"
LOG_DIR = Path.home() / "Library" / "Logs" / "mlx-lm-runtime"
LAUNCH_AGENT_LABEL = "com.mlx-lm.local-runtime"
LAUNCH_AGENT_PATH = (
    Path.home() / "Library" / "LaunchAgents" / (LAUNCH_AGENT_LABEL + ".plist")
)
