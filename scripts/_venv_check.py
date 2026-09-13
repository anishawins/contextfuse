"""Fail with a useful sentence instead of a ModuleNotFoundError."""
import sys
from pathlib import Path


def require_venv() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        venv = root / ".venv" / "bin" / "activate"
        sys.exit(
            "\n!!! The virtual environment is not active.\n"
            f"    Run:  source {venv}\n"
            "    (your shell prompt should start with (.venv) )\n")
