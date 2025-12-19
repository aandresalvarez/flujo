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

    # Ephemeral override: allow config/env to force an in-memory backend
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

    # Default fallback
    if uri is None:
        uri = "sqlite:///flujo_ops.db"
        logging.warning(
            "[flujo.config] FLUJO_STATE_URI not set, using default 'sqlite:///flujo_ops.db'"
        )

    uri_str = uri
    parsed = urlparse(uri_str)
    config_dir = Path.cwd()
    try:
        cfg_path = get_config_manager().config_path
        if cfg_path:
            config_dir = cfg_path.parent
    except Exception:
        pass

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

        # Ensure parent directory exists for SQLite (best effort)
        parent_dir = db_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass  # Backend will fail later if still missing

        return SQLiteBackend(db_path)
    else:
        # Support memory-like URIs when scheme was present but not sqlite
        if (uri or "").strip().lower() in {"memory", "memory://", "mem://", "inmemory://"}:
            return InMemoryBackend()
        print_rich_or_typer(f"[red]Unsupported backend URI: {uri}[/red]", stderr=True)
        raise typer.Exit(1)
