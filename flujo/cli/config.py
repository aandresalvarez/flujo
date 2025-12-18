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
from ..state.sqlite_uri import normalize_sqlite_path as _normalize_sqlite_path
from ..infra.config_manager import get_config_manager, get_state_uri
from ..utils.config import get_settings
from .helpers import print_rich_or_typer


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

    # In test environments, allow opting into an isolated state backend when no explicit
    # FLUJO_STATE_URI env override is provided. This keeps tests hermetic without
    # surprising production runs that may set FLUJO_TEST_MODE for debugging.
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
        # Only override configured state backends when the caller explicitly opts in.
        override_dir = os.getenv("FLUJO_TEST_STATE_DIR", "").strip()

        # Respect explicit ephemeral overrides in tests
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

        if override_dir:
            temp_dir = Path(override_dir).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            return SQLiteBackend(temp_dir / "flujo_ops.db")
        # Fall through to honoring the configured URI.

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

    # Default fallback with optional test isolation
    if uri is None:
        settings = get_settings()
        # Detect explicit test mode to avoid reusing a possibly corrupted repo DB
        is_test_env = settings.test_mode
        override_dir = os.getenv("FLUJO_TEST_STATE_DIR", "").strip() if is_test_env else ""
        if is_test_env and override_dir:
            # Respect explicit test dir exactly (no per-worker subdir)
            temp_dir = Path(override_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{(temp_dir / 'flujo_ops.db').as_posix()}"
            logging.warning(
                f"[flujo.config] FLUJO_STATE_URI not set; using isolated test DB '{uri}'"
            )
            return SQLiteBackend(temp_dir / "flujo_ops.db")

        if is_test_env and not override_dir:
            logging.warning(
                "[flujo.config] FLUJO_TEST_MODE is enabled but FLUJO_TEST_STATE_DIR is not set; "
                "honoring configured/default state backend."
            )
            uri = "sqlite:///flujo_ops.db"

        if not is_test_env:
            logging.warning(
                "[flujo.config] FLUJO_STATE_URI not set, using default 'sqlite:///flujo_ops.db'"
            )
            uri = "sqlite:///flujo_ops.db"

    if uri is None:
        uri = "sqlite:///flujo_ops.db"
    uri_str = uri

    parsed = urlparse(uri_str)
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
            uri_str,
            auto_migrate=auto_migrate,
            pool_min_size=pool_min,
            pool_max_size=pool_max,
        )

    if parsed.scheme.startswith("sqlite"):
        # Env overrides should resolve relative SQLite paths against the current working directory,
        # not against a config file location that may come from a different project.
        sqlite_config_dir = None if env_uri_set else config_dir
        db_path = _normalize_sqlite_path(uri_str, Path.cwd(), config_dir=sqlite_config_dir)
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
