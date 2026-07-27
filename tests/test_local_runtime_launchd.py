import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_lm.local_runtime import launchd


class TestLocalRuntimeLaunchd(unittest.TestCase):
    def test_install_writes_expected_foreground_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent_path = root / "LaunchAgents" / "runtime.plist"
            log_dir = root / "Logs"
            with (
                patch.object(launchd, "LAUNCH_AGENT_PATH", agent_path),
                patch.object(launchd, "LOG_DIR", log_dir),
            ):
                written = launchd.install_launch_agent(
                    "model-path",
                    socket_path=root / "runtime.sock",
                    prompt_cache_size=3,
                    prompt_cache_bytes=1234,
                    idle_timeout=45,
                    start=False,
                )

            self.assertEqual(written, agent_path)
            with agent_path.open("rb") as stream:
                payload = plistlib.load(stream)
            arguments = payload["ProgramArguments"]
            self.assertIn("mlx_lm.local_runtime.cli", arguments)
            self.assertIn("model-path", arguments)
            self.assertIn("1234", arguments)
            self.assertIn("--idle-timeout", arguments)
            self.assertIn("45", arguments)
            self.assertTrue(payload["RunAtLoad"])
            self.assertEqual(payload["KeepAlive"], {"SuccessfulExit": False})


if __name__ == "__main__":
    unittest.main()
