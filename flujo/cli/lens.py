from __future__ import annotations

import typer
import asyncio
from rich.table import Table
from rich.console import Console

from .config import load_backend_from_config

lens_app = typer.Typer(help="Operational inspection commands")


@lens_app.command("list")
def list_runs(
    status: str | None = typer.Option(None),
    pipeline: str | None = typer.Option(None),
    limit: int = typer.Option(50, help="Maximum number of runs to display"),
) -> None:
    """List stored runs."""
    backend = load_backend_from_config()
    try:
        # Use the new structured API if available, fallback to legacy
        if hasattr(backend, "list_runs"):
            runs = asyncio.run(
                backend.list_runs(status=status, pipeline_name=pipeline, limit=limit)
            )
        else:
            runs = asyncio.run(
                backend.list_workflows(status=status, pipeline_id=pipeline, limit=limit)
            )
    except NotImplementedError:
        typer.echo("Backend does not support listing runs", err=True)
        raise typer.Exit(1)

    table = Table("run_id", "pipeline", "status", "created_at")
    for r in runs:
        table.add_row(
            r.get("run_id", "-"),
            r.get("pipeline_name", "-"),
            r.get("status", "-"),
            str(r.get("start_time", "-")),
        )
    Console().print(table)


@lens_app.command("show")
def show_run(run_id: str) -> None:
    """Show detailed information about a run."""
    backend = load_backend_from_config()
    try:
        details = asyncio.run(backend.get_run_details(run_id))
        steps = asyncio.run(backend.list_run_steps(run_id))
    except NotImplementedError:
        typer.echo("Backend does not support run inspection", err=True)
        raise typer.Exit(1)

    if details is None:
        typer.echo("Run not found", err=True)
        raise typer.Exit(1)
    assert details is not None

    table = Table("index", "step", "status")
    for s in steps:
        table.add_row(str(s.get("step_index")), s.get("step_name", "-"), s.get("status", "-"))

    Console().print(f"Run {run_id} - {details['status']}")
    Console().print(table)
