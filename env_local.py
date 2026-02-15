from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """
    Minimal .env loader (no external deps).

    - Ignores empty lines and comments starting with '#'
    - Supports KEY=VALUE (VALUE may be single/double-quoted)
    - Does not overwrite existing environment variables
    """

    try:
        # Unit tests should control env explicitly; don't auto-import secrets from .env.
        if os.getenv("PYTEST_CURRENT_TEST"):
            return
        if os.getenv("DISABLE_DOTENV"):
            return

        p = Path(path)
        if not p.is_absolute():
            # Make .env discovery independent of the current working directory.
            p = (Path(__file__).resolve().parent / p).resolve()
        if not p.exists() or not p.is_file():
            return
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            # Allow .env to populate vars that are unset OR set to an empty string.
            if key not in os.environ or os.environ.get(key, "") == "":
                os.environ[key] = value
    except Exception:
        # Never fail app startup because of a local dev convenience feature.
        return
