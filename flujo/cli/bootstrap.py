"""CLI bootstrap utilities for environment setup and service initialization."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from flujo.infra import telemetry
from .helpers import resolve_project_root


def setup_cli_environment(project_root: Optional[str] = None) -> None:
    """Set up CLI environment including sys.path manipulation for project root.

    Args:
        project_root: Optional explicit project root path. If None, attempts to resolve
            from current directory or FLUJO_PROJECT_ROOT env var.
    """
    try:
        # Ensure project root is importable for custom packages (e.g., skills/)
        if project_root:
            explicit = Path(project_root).resolve()
        else:
            explicit = None

        root = resolve_project_root(explicit=explicit, allow_missing=True)

        if root is not None:
            os.environ["FLUJO_PROJECT_ROOT"] = str(root)

        # Add project root to sys.path if not already present
        if root is not None:
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
        else:
            # Fallback: add current directory if not in sys.path
            cwd_str = str(Path.cwd())
            if cwd_str not in sys.path:
                sys.path.insert(0, cwd_str)
    except Exception:
        # Non-fatal: individual commands may still set defaults or load explicit files
        pass


def initialize_cli_services(register_builtins: bool = True, enable_telemetry: bool = True) -> None:
    """Initialize CLI services including telemetry and builtin skill registration.

    Args:
        register_builtins: If True, imports flujo.builtins to register builtin skills.
        enable_telemetry: If False, skips telemetry initialization (used in tests/daemons).
    """
    # Initialize telemetry at the start of CLI execution
    if enable_telemetry:
        telemetry.init_telemetry()

    # Register builtin skills via import side-effect
    if register_builtins:
        import flujo.builtins  # noqa: F401  # Register builtin skills on CLI import


def configure_cli_logging(
    debug: bool = False,
    verbose: bool = False,
    trace: bool = False,
) -> None:
    """Configure CLI logging based on debug/verbose/trace flags.

    Args:
        debug: Enable debug file logging to '.flujo/logs/run.log'
        verbose: Enable verbose traces for helpers and error handlers
        trace: Alias for verbose to print full Python tracebacks
    """
    # Optional global debug logging to a local file
    if debug:
        try:
            os.makedirs(".flujo/logs", exist_ok=True)
            fh = logging.FileHandler(".flujo/logs/run.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            fh.setFormatter(fmt)
            logger = logging.getLogger("flujo")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        except Exception:
            # Never fail CLI due to logging setup issues
            pass

    # Quiet by default: reduce console noise unless --debug
    try:
        logger = logging.getLogger("flujo")
        if debug:
            # Propagate debug intent to runtime via env for internal warnings gates
            try:
                os.environ["FLUJO_DEBUG"] = "1"
            except Exception:
                pass
            logger.setLevel(logging.INFO)
            for h in list(logger.handlers):
                try:
                    h.setLevel(logging.INFO)
                except Exception:
                    pass
        else:
            # Ensure flag is not set when not debugging
            try:
                if os.environ.get("FLUJO_DEBUG"):
                    del os.environ["FLUJO_DEBUG"]
            except Exception:
                pass
            logger.setLevel(logging.WARNING)
            for h in list(logger.handlers):
                # Keep error handler; downgrade others to WARNING
                try:
                    h.setLevel(logging.WARNING)
                except Exception:
                    pass
    except Exception:
        pass

    # Enable verbose traces for helpers and error handlers
    try:
        if verbose or trace:
            os.environ["FLUJO_CLI_VERBOSE"] = "1"
    except Exception:
        pass


def bootstrap_cli_runtime(
    project_root: Optional[str] = None,
    *,
    register_builtins: bool = True,
    enable_telemetry: bool = True,
) -> None:
    """Centralized runtime bootstrap for CLI and programmatic entrypoints."""

    try:
        from .test_setup import configure_test_environment

        configure_test_environment()
    except Exception:
        # Non-fatal in production environments
        pass

    if enable_telemetry or register_builtins:
        initialize_cli_services(
            register_builtins=register_builtins, enable_telemetry=enable_telemetry
        )
    setup_cli_environment(project_root=project_root)
