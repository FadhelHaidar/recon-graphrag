"""Runtime version and git metadata helpers."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("recon-graphrag")
except PackageNotFoundError:  # pragma: no cover - running from source without install
    __version__ = "0.0.0"


_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_git_sha(fallback: str = "unknown") -> str:
    """Return the current git SHA, or ``fallback`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:  # pragma: no cover - defensive fallback
        return fallback
