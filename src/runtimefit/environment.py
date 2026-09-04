from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from runtimefit import __version__


RUNTIME_COMMANDS = {
    "llama.cpp": ("llama-server", "--version"),
    "ollama": ("ollama", "--version"),
    "vllm": ("vllm", "--version"),
    "nvidia": ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
}


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _command_details(arguments: tuple[str, ...]) -> dict[str, Any]:
    executable = shutil.which(arguments[0])
    if not executable:
        return {"available": False}
    try:
        completed = subprocess.run(
            [executable, *arguments[1:]],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        combined = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        safe_lines = [line for line in combined.splitlines() if line.strip()]
        version_lines = [line for line in safe_lines if "version" in line.casefold()]
        selected_lines = version_lines[:3] if version_lines else safe_lines[-4:]
        return {
            "available": True,
            "path": executable,
            "version_output": " | ".join(selected_lines)[:500] or "unknown",
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "path": executable, "version_output": f"unavailable: {type(exc).__name__}"}


def collect_environment() -> dict[str, Any]:
    return {
        "runtimefit": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "tools": {name: _command_details(command) for name, command in RUNTIME_COMMANDS.items()},
    }
