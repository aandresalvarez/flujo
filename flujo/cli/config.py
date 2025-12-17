from __future__ import annotations

import os
import importlib.util
from pathlib import Path
from urllib.parse import urlparse
import logging

# Configure logging: show DEBUG if FLUJO_DEBUG=1, else INFO
if os.getenv("FLUJO_DEBUG") == "1":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

from ..state.backends.sqlite import SQLiteBackend
from ..state.backends.postgres import PostgresBackend
from ..state.backends.memory import InMemoryBackend
from ..state.backends.base import StateBackend
from ..infra.config_manager import get_config_manager, get_state_uri
from ..utils.config import get_settings
import tempfile
from .helpers import print_rich_or_typer


def _normalize_sqlite_path(uri: str, cwd: Path, *, config_dir: Path | None = None) -> Path:
    """
    Normalize a SQLite URI path to an absolute or relative Path.

    - If the path is absolute (e.g., sqlite:////abs/path.db), return as is.
    - If the path is relative (e.g., sqlite:///foo.db), resolve relative to cwd (or config_dir).
    - Handles Windows paths and non-standard URIs gracefully.
    """
    parsed = urlparse(uri)
    base_dir = config_dir if config_dir is not None else cwd

    # Case 1: Non-standard sqlite://path (netloc present) or Windows drive in netloc
    if parsed.netloc:
        path_str = parsed.netloc + parsed.path
        # Windows drive logic (C:/... or C:\...)
        try:
            if ":" in path_str and Path(path_str).is_absolute():
                return Path(path_str).resolve()
        except Exception:
            pass

        # If it looks like an absolute path, treat as such
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()

        return (base_dir / path_str).resolve()

    # Case 2: Standard URI path
    path_str = parsed.path
    if not path_str or not path_str.strip():
        raise ValueError(
            "Malformed SQLite URI: empty path. Use 'sqlite:///file.db' or 'sqlite:////abs/path.db'."
        )

    # Check for // implying absolute path (sqlite:////abs...)
    if path_str.startswith("//"):
        return Path(path_str[1:]).resolve()

    # Strip standard leading slash for processing
    clean_path_str = path_str[1:] if path_str.startswith("/") else path_str

    # Check if the remaining part is absolute (e.g. Windows /C:/...)
    try:
        p = Path(clean_path_str)
        if p.is_absolute():
            return p.resolve()
    except Exception:
        pass

    # Default to specific behavior: sqlite:///foo.db is relative
    return (base_dir / clean_path_str).resolve()


def load_backend_from_config() -> StateBackend:
    """Load a state backend based on configuration, with robust error handling."""
    import typer

    # Fast path: when the user explicitly set FLUJO_STATE_URI, don't force a ConfigManager
    # reload on every CLI invocation (important for stable perf in CI and perf tests).
    env_uri = os.getenv("FLUJO_STATE_URI", "").strip()
    if env_uri:
        uri: str | None = env_uri
        env_uri_set = True
    else:
        # Handles env vars + TOML with proper precedence; force_reload keeps interactive
        # sessions consistent when the config file changes on disk.
        uri = get_state_uri(force_reload=True)
        env_uri_set = False

    # In test environments, prefer an isolated temp SQLite DB when no explicit
    # FLUJO_STATE_URI env override is provided — even if flujo.toml specifies
    # a state_uri. This keeps tests hermetic and avoids writing to the repo db.
    if not env_uri_set:
        try:
            settings = get_settings()
            # Strict Context Hygiene: Relies solely on settings.test_mode (driven by FLUJO_TEST_MODE)
            is_test_env = settings.test_mode
        except Exception:
            is_test_env = False
    else:
        is_test_env = False
    if is_test_env and not env_uri_set:
        # Respect ephemeral overrides in tests
        try:
            mode = os.getenv("FLUJO_STATE_MODE", "").strip().lower()
            ephemeral = os.getenv("FLUJO_EPHEMERAL_STATE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            mode, ephemeral = "", False
        if mode in {"memory", "ephemeral"} or ephemeral:
            return InMemoryBackend()

        override_dir = os.getenv("FLUJO_TEST_STATE_DIR")
        if override_dir and override_dir.strip():
            temp_dir = Path(override_dir.strip()).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            return SQLiteBackend(temp_dir / "flujo_ops.db")
        else:
            # Fallback to system temp for isolated test DBs
            base_dir = Path(os.getenv("PYTEST_TMPDIR", tempfile.gettempdir())) / "flujo-test-db"
            try:
                worker_id = os.getenv("PYTEST_XDIST_WORKER", "")
            except Exception:
                worker_id = ""
            try:
                pid = os.getpid()
            except Exception:
                pid = 0
            subdir = f"worker-{worker_id or 'single'}-pid-{pid}"
            temp_dir = base_dir / subdir
            temp_dir.mkdir(parents=True, exist_ok=True)
            return SQLiteBackend(temp_dir / "flujo_ops.db")

    # Ephemeral override: allow config/env to force an in-memory backend
    # Accepted forms:
    #   - FLUJO_STATE_URI=memory:// (or mem://, inmemory://, memory)
    #   - FLUJO_STATE_MODE in {"memory", "ephemeral"}
    #   - FLUJO_EPHEMERAL_STATE in truthy values
    try:
        mode = os.getenv("FLUJO_STATE_MODE", "").strip().lower()
        ephemeral_flag = os.getenv("FLUJO_EPHEMERAL_STATE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        uri_lower = (uri or "").strip().lower()
        memory_uri = uri_lower in {"memory", "memory://", "mem://", "inmemory://"}
        if mode in {"memory", "ephemeral"} or ephemeral_flag or memory_uri:
            return InMemoryBackend()
    except Exception:
        # Fall through to normal resolution
        pass

    # Default fallback with test/CI isolation
    if uri is None:
        settings = get_settings()
        # Detect explicit test mode to avoid reusing a possibly corrupted repo DB
        is_test_env = settings.test_mode
        if is_test_env:
            # Allow override for test DB directory
            override_dir = os.getenv("FLUJO_TEST_STATE_DIR")
            if override_dir and override_dir.strip():
                # Respect explicit test dir exactly (no per-worker subdir)
                temp_dir = Path(override_dir.strip())
                temp_dir.mkdir(parents=True, exist_ok=True)
                uri = f"sqlite:///{(temp_dir / 'flujo_ops.db').as_posix()}"
                logging.warning(
                    f"[flujo.config] FLUJO_STATE_URI not set; using isolated test DB '{uri}'"
                )
                # Return early since we've constructed the final URI
                return SQLiteBackend(temp_dir / "flujo_ops.db")
            else:
                # Base directory under system temp
                base_dir = Path(os.getenv("PYTEST_TMPDIR", tempfile.gettempdir())) / "flujo-test-db"
            # Create a per-worker/per-process subdirectory to avoid cross-test collisions
            try:
                worker_id = os.getenv("PYTEST_XDIST_WORKER", "")
            except Exception:
                worker_id = ""
            try:
                pid = os.getpid()
            except Exception:
                pid = 0
            subdir = f"worker-{worker_id or 'single'}-pid-{pid}"
            temp_dir = base_dir / subdir
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
    config_path = None
    try:
        cfg_mgr = get_config_manager(force_reload=False)
        config_path = getattr(cfg_mgr, "config_path", None)
    except Exception:
        config_path = None
    config_dir = config_path.parent if config_path is not None else Path.cwd()

    if parsed.scheme.lower() in {"postgres", "postgresql"}:
        # Guard: Check if asyncpg is available before creating PostgresBackend
        spec = importlib.util.find_spec("asyncpg")
        if spec is None:
            print_rich_or_typer(
                "[red]Error: asyncpg is required for PostgreSQL support. "
                "Install with `pip install flujo[postgres]`.[/red]",
                stderr=True,
            )
            raise typer.Exit(1)
        cfg_manager = get_config_manager()
        settings_model = cfg_manager.get_settings()
        pool_min = getattr(settings_model, "postgres_pool_min", 1)
        pool_max = getattr(settings_model, "postgres_pool_max", 10)
        auto_migrate = os.getenv("FLUJO_AUTO_MIGRATE", "true").lower() != "false"
        return PostgresBackend(
            uri,
            auto_migrate=auto_migrate,
            pool_min_size=pool_min,
            pool_max_size=pool_max,
        )

    if parsed.scheme.startswith("sqlite"):
        # Env overrides should resolve relative SQLite paths against the current working directory,
        # not against a config file location that may come from a different project.
        sqlite_config_dir = None if env_uri_set else config_dir
        db_path = _normalize_sqlite_path(uri, Path.cwd(), config_dir=sqlite_config_dir)
        # Debug output for test visibility
        if os.getenv("FLUJO_DEBUG") == "1":
            logging.debug(f"[flujo.config] Using SQLite DB path: {db_path}")
        parent_dir = db_path.parent
        # Fast path in tests/CI: skip heavy permission checks to keep CLI latency low
        try:
            from ..utils.config import get_settings as _get_settings

            _settings = _get_settings()
            _is_test_env = _settings.test_mode
        except Exception:
            _is_test_env = False

        if _is_test_env:
            # Assume fixture created the database; just return the backend
            return SQLiteBackend(db_path)

        # Do NOT auto-create parent directories; fail if missing
        if not parent_dir.exists():
            print_rich_or_typer(
                f"[red]Error: Database directory '{parent_dir}' does not exist[/red]",
                stderr=True,
            )
            raise typer.Exit(1)
        if not os.access(parent_dir, os.W_OK):
            print_rich_or_typer(
                f"[red]Error: Database directory '{parent_dir}' is not writable[/red]",
                stderr=True,
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
            print_rich_or_typer(
                f"[red]Error: Cannot create/open database file '{db_path}' with secure permissions due to {type(e).__name__}: {e}[/red]",
                stderr=True,
            )
            raise typer.Exit(1) from e
        if not os.access(db_path, os.W_OK):
            print_rich_or_typer(
                f"[red]Error: Database file '{db_path}' is not writable[/red]",
                stderr=True,
            )
            raise typer.Exit(1)
        # Try to open the file for writing (touch)
        try:
            with open(db_path, "a"):
                pass
        except Exception as e:
            print_rich_or_typer(
                f"[red]Error: Cannot write to database file '{db_path}' due to {type(e).__name__}: {e}[/red]",
                stderr=True,
            )
            raise typer.Exit(1) from e
        return SQLiteBackend(db_path)
    else:
        # Support memory-like URIs when scheme was present but not sqlite
        if (uri or "").strip().lower() in {"memory", "memory://", "mem://", "inmemory://"}:
            return InMemoryBackend()
        print_rich_or_typer(f"[red]Unsupported backend URI: {uri}[/red]", stderr=True)
        raise typer.Exit(1)
