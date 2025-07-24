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
    - Special handling: sqlite:///foo.db (parsed.path == '/foo.db') is relative, not absolute.
    """
    parsed = urlparse(uri)
    path_str = parsed.path

    # RFC 3986: single leading slash after scheme is relative, double is absolute
    # sqlite:///foo.db -> /foo.db (should be relative: foo.db)
    # sqlite:////abs/path.db -> //abs/path.db (should be /abs/path.db, absolute)
    if path_str.startswith("/./"):
        path_str = path_str[1:]  # becomes ./foo.db
    elif path_str.startswith("//"):
        # sqlite:////abs/path.db -> //abs/path.db (should be /abs/path.db)
        path_str = path_str[1:]  # Remove one slash, keep the absolute path
    elif path_str.startswith("/") and not path_str.startswith("//"):
        # sqlite:///foo.db -> /foo.db (should be relative: foo.db)
        # BUT: if the path after removing / is actually an absolute path, keep it absolute
        potential_absolute = path_str[1:]  # Remove leading slash
        # Check if this looks like an absolute path on Unix systems
        if potential_absolute.startswith("/") or potential_absolute.startswith(
            ("private/", "var/", "usr/", "etc/", "tmp/", "home/", "opt/", "sbin/", "bin/")
        ):
            # This is actually an absolute path, prepend / to make it proper absolute
            path_str = "/" + potential_absolute
        else:
            # This is a relative path
            path_str = potential_absolute

    # Now, if path_str is absolute, return as is; else, resolve relative to cwd
    path = Path(path_str)
    if path.is_absolute():
        # Path is already absolute, return as is
        pass
    else:
        # Path is relative, resolve relative to cwd
        path = cwd / path

    logging.debug(f"[flujo.config] Resolved SQLite DB path: {path}")
    return path


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on env var or flujo.toml, with robust error handling."""
    import sys
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
        uri = "sqlite:///flujo_ops.db"
        if not used_env:
            print(
                "[flujo.config] Warning: FLUJO_STATE_URI not set, using default 'sqlite:///flujo_ops.db'",
                file=sys.stderr,
            )

    parsed = urlparse(uri)
    if parsed.scheme.startswith("sqlite"):
        db_path = _normalize_sqlite_path(uri, Path.cwd())
        # Debug output for test visibility
        if os.getenv("FLUJO_DEBUG") == "1":
            print(f"[flujo.config] Using SQLite DB path: {db_path}", file=sys.stderr)
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
