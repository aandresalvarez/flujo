"""Programmatic entrypoint for non-CLI hosts to bootstrap and use Flujo.

This module provides documented APIs for services, tests, and daemons that need
to initialize Flujo without going through the CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

from .app_registration import create_cli_app, register_all_commands
from .bootstrap import bootstrap_cli_runtime


def bootstrap_flujo(
    project_root: Optional[str] = None,
    enable_telemetry: bool = True,
    register_builtins: bool = True,
) -> None:
    """Bootstrap Flujo environment for non-CLI hosts.

    This function initializes the Flujo runtime environment including:
    - Project root resolution and sys.path setup
    - Telemetry initialization (if enabled)
    - Builtin skill registration (if enabled)

    This is the recommended entrypoint for services, tests, and daemons that
    need to use Flujo programmatically without going through the CLI.

    Args:
        project_root: Optional explicit project root path. If None, attempts to
            resolve from current directory or FLUJO_PROJECT_ROOT env var.
        enable_telemetry: If True, initializes telemetry services. Set to False
            for minimal initialization in test environments.
        register_builtins: If True, registers builtin Flujo skills. Set to False
            if you only need core pipeline execution without builtin skills.

    Example:
        ```python
        from flujo.cli.programmatic import bootstrap_flujo

        # Initialize Flujo for a web service
        bootstrap_flujo(project_root="/path/to/project")

        # Now you can use Flujo APIs
        from flujo import Flujo
        runner = Flujo(pipeline=my_pipeline)
        result = runner.run("input data")
        ```

    Example for tests:
        ```python
        from flujo.cli.programmatic import bootstrap_flujo

        # Minimal bootstrap for tests (no telemetry, no builtins)
        bootstrap_flujo(enable_telemetry=False, register_builtins=False)
        ```
    """
    bootstrap_cli_runtime(
        project_root=project_root,
        register_builtins=register_builtins,
        enable_telemetry=enable_telemetry,
    )


def create_flujo_app() -> typer.Typer:
    """Create a fully configured Flujo CLI app without executing it.

    This function returns a configured Typer app with all commands registered,
    suitable for embedding in custom Typer applications or for programmatic
    CLI execution.

    Returns:
        Fully configured Typer app with all Flujo commands registered.

    Example:
        ```python
        from flujo.cli.programmatic import create_flujo_app
        import typer

        # Create Flujo app
        flujo_app = create_flujo_app()

        # Embed in custom app
        my_app = typer.Typer()
        my_app.add_typer(flujo_app, name="flujo")

        # Or use directly
        if __name__ == "__main__":
            flujo_app()
        ```

    Note:
        This function does NOT initialize telemetry or register builtins.
        Call `bootstrap_flujo()` first if you need those services.
    """
    app = create_cli_app()
    register_all_commands(app)
    return app
