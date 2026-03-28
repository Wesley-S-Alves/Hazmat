"""Platform compatibility utilities.

Handles OpenMP thread limits, device detection, and path resolution
across macOS Apple Silicon, Linux, and Docker environments.
"""

import os
import platform
import sys


def _is_apple_silicon() -> bool:
    """Check if running on macOS with Apple Silicon (ARM)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def configure_omp() -> None:
    """Configure OpenMP for the current platform.

    On macOS Apple Silicon, XGBoost/LightGBM segfault with multi-threaded OpenMP.
    This must be called BEFORE importing xgboost, lightgbm, or sklearn.
    On Linux/Docker, multi-threading is safe.
    """
    if _is_apple_silicon():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")


def get_project_root():
    """Get the project root directory (where pyproject.toml lives)."""
    from pathlib import Path

    # Try relative to this file (src/compat.py -> project root)
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists():
        return root

    # Fallback to CWD
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd

    # Last resort: env var
    env_root = os.environ.get("HAZMAT_PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    return cwd


# Auto-configure OMP on import (safe to import early)
configure_omp()
