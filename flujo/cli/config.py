from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from ..state.backends.sqlite import SQLiteBackend
from ..state.backends.base import StateBackend
from ..infra.config_manager import get_state_uri


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on env var or flujo.toml."""
    # First check environment variable
    uri = os.getenv("FLUJO_STATE_URI")

    # If not found, try to get from configuration file
    if uri is None:
        uri = get_state_uri()

    # Default fallback
    if uri is None:
        uri = "sqlite:///flujo_ops.db"

    parsed = urlparse(uri)
    if parsed.scheme.startswith("sqlite"):
        # Handle absolute paths correctly
        # sqlite:///path -> path should be /path, not //path
        path_str = parsed.path
        if path_str.startswith("//"):
            path_str = path_str[1:]  # Remove the extra slash
        path = Path(path_str or "flujo_ops.db")
        return SQLiteBackend(path)
    raise ValueError(f"Unsupported backend URI: {uri}")
