from __future__ import annotations

import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from typing import Any

from runtimefit.config import ConfigError


class ManagedProcess(AbstractContextManager["ManagedProcess"]):
    def __init__(self, specification: dict[str, Any]):
        command = specification.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise ConfigError("launch.command must be a non-empty list of strings")
        self.command = [os.path.expandvars(part) for part in command]
        if any("${" in part for part in self.command):
            raise ConfigError("launch.command contains an unset environment variable")
        self.health_url = str(specification.get("health_url", ""))
        if not self.health_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ConfigError("launch.health_url must use localhost")
        self.startup_timeout_s = float(specification.get("startup_timeout_s", 120))
        self.process: subprocess.Popen[bytes] | None = None
        self.startup_seconds: float | None = None
        self._log = tempfile.TemporaryFile()

    def __enter__(self) -> "ManagedProcess":
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    detail = self._log_tail()
                    raise ConfigError(
                        f"Managed process exited with code {self.process.returncode} before becoming healthy:\n{detail}"
                    )
                try:
                    with urllib.request.urlopen(self.health_url, timeout=1) as response:
                        if 200 <= response.status < 500:
                            self.startup_seconds = time.perf_counter() - started
                            return self
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.25)
            raise ConfigError(f"Managed process did not become healthy within {self.startup_timeout_s:g}s")
        except Exception:
            self._stop()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop()

    def _stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if not self._log.closed:
            self._log.close()

    def _log_tail(self) -> str:
        self._log.flush()
        self._log.seek(0)
        text = self._log.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-20:])
