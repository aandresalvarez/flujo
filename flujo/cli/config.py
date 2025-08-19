from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
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
from ..utils.config import get_settings
import tempfile


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
        # Handle Windows drive letter like C:/path or C:\path
        try:
            if ":" in path_str or path_str.startswith("\\"):
                win = PureWindowsPath(path_str)
                if win.is_absolute():
                    return Path(str(win))
        except Exception:
            pass
        if path_str.startswith("/"):
            return Path(path_str).resolve()
        return (cwd / path_str).resolve()
    # Case 2: Standard sqlite:///foo.db (netloc empty, path present)
    elif not parsed.netloc and parsed.path:
        path_str = parsed.path
        # If path_str starts with '//', treat as absolute (sqlite:////abs/path.db)
        if path_str.startswith("//"):
            return Path(path_str[1:]).resolve()  # Remove one slash to get /abs/path.db
        # Windows absolute path like /C:/path or C:/path
        try:
            normalized = path_str[1:] if path_str.startswith("/") else path_str
            win = PureWindowsPath(normalized)
            if win.is_absolute():
                return Path(str(win))
        except Exception:
            pass
        # Otherwise treat as relative to cwd
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        return (cwd / rel_path).resolve()
    # Case 3: netloc and path both present (rare, but possible)
    elif parsed.netloc:
        path_str = (
            f"/{parsed.netloc}{parsed.path}"
            if not parsed.path.startswith("/")
            else f"{parsed.netloc}{parsed.path}"
        )
        # If path_str starts with '//', treat as absolute
        if path_str.startswith("//"):
            return Path(path_str[1:]).resolve()
        # Windows absolute path
        try:
            normalized = path_str[1:] if path_str.startswith("/") else path_str
            win = PureWindowsPath(normalized)
            if win.is_absolute():
                return Path(str(win))
        except Exception:
            pass
        # Otherwise, treat as relative
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        return (cwd / rel_path).resolve()
    else:
        # Fallback: treat as relative, but guard against empty path (e.g., sqlite://)
        path_str = parsed.path
        if not parsed.netloc and (path_str is None or path_str.strip() == ""):
            raise ValueError(
                "Malformed SQLite URI: empty path. Use 'sqlite:///file.db' or 'sqlite:////abs/path.db'."
            )
        rel_path = path_str[1:] if path_str.startswith("/") else path_str
        resolved = (cwd / rel_path).resolve()
        return resolved


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on configuration, with robust error handling."""
    import typer

    # Get state URI from ConfigManager (handles env vars + TOML with proper precedence)
    uri = get_state_uri(force_reload=True)

    # Default fallback with test/CI isolation
    if uri is None:
        settings = get_settings()
        # Detect pytest or explicit test mode to avoid reusing a possibly corrupted repo DB
        is_test_env = bool(os.getenv("PYTEST_CURRENT_TEST")) or settings.test_mode
        if is_test_env:
            # Allow override for test DB directory
            override_dir = os.getenv("FLUJO_TEST_STATE_DIR")
            if override_dir and override_dir.strip():
                temp_dir = Path(override_dir.strip())
            else:
                # Place DB in a temp directory unique per user/session to avoid collisions
                temp_dir = Path(os.getenv("PYTEST_TMPDIR", tempfile.gettempdir())) / "flujo-test-db"
            temp_dir.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{(temp_dir / 'flujo_ops.db').as_posix()}"
            logging.warning(
                f"[flujo.config] FLUJO_STATE_URI not set; using isolated test DB '{uri}'"
            )
        else:
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
                f"[red]Error: Database directory '{parent_dir}' does not exist[/red]",
                err=True,
            )
            raise typer.Exit(1)
        if not os.access(parent_dir, os.W_OK):
            typer.echo(
                f"[red]Error: Database directory '{parent_dir}' is not writable[/red]",
                err=True,
            )
            raise typer.Exit(1)
        # Ensure the database file exists with secure permissions (read/write for owner only)
        # This avoids issues with default umask or inherited permissions from 'open' in append mode.
        try:
            # Atomically create the DB file with secure permissions when absent, or open if present
            flags = os.O_CREAT | os.O_WRONLY
            fd = os.open(db_path, flags, 0o600)
            try:
                # Ensure permissions are set to 0600 even if file pre-existed with looser perms
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
        except Exception as e:
            typer.echo(
                f"[red]Error: Cannot create/open database file '{db_path}' with secure permissions due to {type(e).__name__}: {e}[/red]",
                err=True,
            )
            raise typer.Exit(1)
        if not os.access(db_path, os.W_OK):
            typer.echo(f"[red]Error: Database file '{db_path}' is not writable[/red]", err=True)
            raise typer.Exit(1)
        # Try to open the file for writing (touch)
        try:
            with open(db_path, "a"):
                pass
        except Exception as e:
            typer.echo(
                f"[red]Error: Cannot write to database file '{db_path}' due to {type(e).__name__}: {e}[/red]",
                err=True,
            )
            raise typer.Exit(1)
        return SQLiteBackend(db_path)
    else:
        typer.echo(f"[red]Unsupported backend URI: {uri}[/red]", err=True)
        raise typer.Exit(1)
