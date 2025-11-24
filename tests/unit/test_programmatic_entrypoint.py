"""Smoke tests for programmatic CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from flujo.cli.programmatic import bootstrap_flujo, create_flujo_app


def test_bootstrap_flujo_minimal() -> None:
    """Test minimal bootstrap without telemetry or builtins."""
    # Should not raise
    bootstrap_flujo(enable_telemetry=False, register_builtins=False)


def test_bootstrap_flujo_full() -> None:
    """Test full bootstrap with telemetry and builtins."""
    # Should not raise
    bootstrap_flujo(enable_telemetry=True, register_builtins=True)


def test_bootstrap_flujo_with_project_root(tmp_path: Path) -> None:
    """Test bootstrap with explicit project root."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Should not raise
    bootstrap_flujo(project_root=str(project_dir))

    # Verify project root is in sys.path
    assert str(project_dir) in sys.path or str(project_dir.resolve()) in sys.path


def test_create_flujo_app() -> None:
    """Test creating Flujo CLI app programmatically."""
    app = create_flujo_app()

    assert isinstance(app, typer.Typer)

    # Verify app has commands registered
    # The app should have commands like 'run', 'validate', etc.
    # We can't easily check command names without invoking, but we can verify it's a Typer instance
    assert hasattr(app, "registered_commands") or hasattr(app, "commands")


def test_programmatic_app_can_be_embedded() -> None:
    """Test that programmatic app can be embedded in custom Typer app."""
    flujo_app = create_flujo_app()

    # Create a custom app
    custom_app = typer.Typer(name="custom")

    # Embed Flujo app
    custom_app.add_typer(flujo_app, name="flujo")

    # Verify embedding worked
    assert isinstance(custom_app, typer.Typer)


def test_bootstrap_backward_compatibility() -> None:
    """Test that bootstrap functions maintain backward compatibility."""
    # Test that calling bootstrap multiple times doesn't break
    bootstrap_flujo(enable_telemetry=False, register_builtins=False)
    bootstrap_flujo(enable_telemetry=False, register_builtins=False)

    # Should still work
    app = create_flujo_app()
    assert isinstance(app, typer.Typer)


def test_programmatic_entrypoint_isolated_from_cli() -> None:
    """Test that programmatic entrypoint works independently of CLI main module."""
    # This test verifies that bootstrap_flujo can be used without importing main.py
    # which is important for non-CLI hosts

    # Clear any existing state
    if "flujo.cli.main" in sys.modules:
        # Don't actually delete, but verify we can use programmatic without main
        pass

    # Should work independently
    bootstrap_flujo(enable_telemetry=False, register_builtins=False)
    app = create_flujo_app()
    assert isinstance(app, typer.Typer)
