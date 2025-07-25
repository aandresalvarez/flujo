from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
import logging

# Configure logging: show DEBUG if FLUJO_DEBUG=1, else INFO
if os.getenv("FLUJO_DEBUG") == "1":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

from ..state.backends.sqlite import SQLiteBackend
from ..state.backends.base import StateBackend
from ..infra.config_manager import get_state_uri


def _normalize_sqlite_path(uri: str, cwd: Path) -> Path:
    """
    Normalize a SQLite URI path to an absolute or relative Path.

    - If the path is absolute (e.g., sqlite:////abs/path.db or sqlite://{abs_path}), return as is.
    - If the path is relative (e.g., sqlite:///foo.db or sqlite:///./foo.db), resolve relative to cwd.
    - Handles all RFC 3986-compliant forms and SQLite URI variants, including non-standard sqlite://{db_path}.
    """

    parsed = urlparse(uri)
    # Case 1: Non-standard sqlite://{db_path} (netloc present, path empty)
    if parsed.netloc and not parsed.path:
        logging.warning(
            f"Non-standard SQLite URI: '{uri}'. Use 'sqlite:///foo.db' or 'sqlite:////abs/path.db' for portability."
        )
        path_str = parsed.netloc
        if path_str.startswith("/"):
            resolved = Path(path_str).resolve()
            return resolved
        else:
            resolved = (cwd / path_str).resolve()
            return resolved
    # Case 2: Standard sqlite:///foo.db (netloc empty, path present)
    elif not parsed.netloc and parsed.path:
        path_str = parsed.path
        # If path_str starts with '//', treat as absolute (sqlite:////abs/path.db)
        if path_str.startswith("//"):
            resolved = Path(path_str[1:]).resolve()  # Remove one slash to get /abs/path.db
            return resolved
        # For all other cases, always treat as relative (even if starts with '/')
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        resolved = (cwd / rel_path).resolve()
        return resolved
    # Case 3: netloc and path both present (rare, but possible)
    elif parsed.netloc:
        path_str = (
            f"/{parsed.netloc}{parsed.path}"
            if not parsed.path.startswith("/")
            else f"{parsed.netloc}{parsed.path}"
        )
        # If path_str starts with '//', treat as absolute
        if path_str.startswith("//"):
            resolved = Path(path_str[1:]).resolve()
            return resolved
        # Otherwise, treat as relative
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        resolved = (cwd / rel_path).resolve()
        return resolved
    else:
        # Fallback: treat as relative
        path_str = parsed.path
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        resolved = (cwd / rel_path).resolve()
        return resolved


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on env var or flujo.toml, with robust error handling."""
    import typer

    # First check environment variable
    uri = os.getenv("FLUJO_STATE_URI")
    used_env = True

    # If not found, try to get from configuration file
    if uri is None:
        uri = get_state_uri(force_reload=True)
        used_env = False

    # Default fallback
    if uri is None:
        if not used_env:
            logging.warning(
                "[flujo.config] FLUJO_STATE_URI not set, using default 'sqlite:///flujo_ops.db'"
            )
        uri = "sqlite:///flujo_ops.db"

    parsed = urlparse(uri)
    if parsed.scheme.startswith("sqlite"):
        db_path = _normalize_sqlite_path(uri, Path.cwd())
        # Debug output for test visibility
        if os.getenv("FLUJO_DEBUG") == "1":
            logging.debug(f"[flujo.config] Using SQLite DB path: {db_path}")
        parent_dir = db_path.parent
        # Do NOT auto-create parent directories; fail if missing
        if not parent_dir.exists():
            typer.echo(
                f"[red]Error: Database directory '{parent_dir}' does not exist[/red]", err=True
            )
            raise typer.Exit(1)
        if not os.access(parent_dir, os.W_OK):
            typer.echo(
                f"[red]Error: Database directory '{parent_dir}' is not writable[/red]", err=True
            )
            raise typer.Exit(1)
        # Try to open the file for writing (touch)
        try:
            with open(db_path, "a"):
                pass
        except Exception as e:
            typer.echo(
                f"[red]Error: Cannot write to database file '{db_path}': {e}[/red]", err=True
            )
            raise typer.Exit(1)
        return SQLiteBackend(db_path)
    else:
        typer.echo(f"[red]Unsupported backend URI: {uri}[/red]", err=True)
        raise typer.Exit(1)
