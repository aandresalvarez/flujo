from __future__ import annotations


import typer
from typing_extensions import Annotated

from .helpers import scaffold_project, scaffold_demo_project


def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help=(
                "Re-initialize templates even if Flujo project files already exist "
                "(flujo.toml, pipeline.yaml, skills/*)."
            ),
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts when using --force",
        ),
    ] = False,
) -> None:
    """Initialize a new Flujo project in the current directory."""
    try:
        from pathlib import Path as _Path

        if force:
            if not yes:
                proceed = typer.confirm(
                    "This directory already has Flujo project files. Re-initialize templates (overwrite flujo.toml, pipeline.yaml, and skills/*)?",
                    default=False,
                )
                if not proceed:
                    raise typer.Exit(0)
            scaffold_project(_Path.cwd(), overwrite_existing=True)
        else:
            scaffold_project(_Path.cwd())
    except Exception as e:
        typer.secho(f"Failed to initialize project: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


def demo(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help=("Scaffold the demo project even if the directory already contains Flujo files."),
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompts when using --force",
        ),
    ] = False,
) -> None:
    """Creates a new Flujo demo project in the current directory."""
    try:
        from pathlib import Path as _Path

        if force:
            if not yes:
                proceed = typer.confirm(
                    "This directory may already contain a Flujo project. Re-scaffold with demo files?",
                    default=False,
                )
                if not proceed:
                    raise typer.Exit(0)
            scaffold_demo_project(_Path.cwd(), overwrite_existing=True)
        else:
            scaffold_demo_project(_Path.cwd())
    except Exception as e:
        typer.secho(f"Failed to create demo project: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


__all__ = ["init", "demo"]
