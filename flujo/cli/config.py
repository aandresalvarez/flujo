from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
import logging

from ..state.backends.sqlite import SQLiteBackend
from ..state.backends.base import StateBackend
from ..infra.config_manager import get_state_uri


def _normalize_sqlite_path(uri: str, cwd: Path) -> Path:
    """
    Normalize a SQLite URI path to an absolute or relative Path.

    - If the path is absolute (e.g., sqlite:////abs/path.db), return as is.
    - If the path is relative (e.g., sqlite:///./foo.db or sqlite:///foo.db),
      resolve relative to cwd.
    - Handles all RFC 3986-compliant forms.
    """
    parsed = urlparse(uri)
    path_str = parsed.path
    # Remove leading slash for relative paths (sqlite:///./foo.db -> /./foo.db)
    if path_str.startswith("/./"):
        path_str = path_str[1:]  # becomes ./foo.db
    elif path_str.startswith("//"):
        # sqlite:////abs/path.db -> //abs/path.db (should be /abs/path.db)
        path_str = path_str[1:]
    # Now, if path_str is absolute, return as is; else, resolve relative to cwd
    path = Path(path_str)
    if not path.is_absolute():
        path = cwd / path
    logging.debug(f"[flujo.config] Resolved SQLite DB path: {path}")
    return path


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on env var or flujo.toml."""
    # First check environment variable
    uri = os.getenv("FLUJO_STATE_URI")

    # If not found, try to get from configuration file
    if uri is None:
        uri = get_state_uri(force_reload=True)

    # Default fallback
    if uri is None:
        uri = "sqlite:///flujo_ops.db"

    parsed = urlparse(uri)
    if parsed.scheme.startswith("sqlite"):
        db_path = _normalize_sqlite_path(uri, Path.cwd())
        return SQLiteBackend(db_path)
    raise ValueError(f"Unsupported backend URI: {uri}")
