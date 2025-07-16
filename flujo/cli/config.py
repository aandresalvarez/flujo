from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from ..state.backends.sqlite import SQLiteBackend
from ..state.backends.base import StateBackend


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on env var or flujo.toml."""
    uri = os.getenv("FLUJO_STATE_URI")
    if uri is None and Path("flujo.toml").exists():
        import tomllib

        with open("flujo.toml", "rb") as f:
            data = tomllib.load(f)
        uri = data.get("state_uri")

    if uri is None:
        uri = "sqlite:///flujo_ops.db"

    parsed = urlparse(uri)
    if parsed.scheme.startswith("sqlite"):
        path = Path(parsed.path or "flujo_ops.db")
        return SQLiteBackend(path)
    raise ValueError(f"Unsupported backend URI: {uri}")
