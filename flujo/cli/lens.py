import typer
import asyncio
from typing import Optional
from rich.table import Table
from rich.console import Console
from .config import load_backend_from_config
from .lens_show import show_run
from .lens_trace import trace_command

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
def show_command(
    run_id: str,
    show_output: bool = typer.Option(False, "--show-output", help="Show step outputs."),
    show_input: bool = typer.Option(False, "--show-input", help="Show step inputs."),
    show_error: bool = typer.Option(False, "--show-error", help="Show step errors."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show input, output, and error for each step."
    ),
) -> None:
    show_run(
        run_id,
        show_output=show_output,
        show_input=show_input,
        show_error=show_error,
        verbose=verbose,
    )


@lens_app.command("trace")
def trace_command_cli(run_id: str) -> None:
    trace_command(run_id)


@lens_app.command("spans")
def list_spans(
    run_id: str,
    status: Optional[str] = typer.Option(None, help="Filter by span status"),
    name: Optional[str] = typer.Option(None, help="Filter by span name"),
) -> None:
    """List individual spans for a run with optional filtering."""
    backend = load_backend_from_config()
    try:
        spans = asyncio.run(backend.get_spans(run_id, status=status, name=name))
    except NotImplementedError:
        typer.echo("Backend does not support span-level querying", err=True)
        raise typer.Exit(1)

    if not spans:
        typer.echo(f"No spans found for run_id: {run_id}")
        return

    table = Table("span_id", "name", "status", "start_time", "end_time", "duration", "parent")
    for span in spans:
        start_time = span.get("start_time")
        end_time = span.get("end_time")
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = f"{float(end_time) - float(start_time):.2f}s"
            except Exception:
                duration = "N/A"
        else:
            duration = "N/A"

        table.add_row(
            span.get("span_id", "-")[:8] + "...",  # Truncate long IDs
            span.get("name", "-"),
            span.get("status", "-"),
            str(start_time) if start_time else "-",
            str(end_time) if end_time else "-",
            duration,
            span.get("parent_span_id", "-")[:8] + "..." if span.get("parent_span_id") else "-",
        )

    Console().print(f"Spans for run {run_id}:")
    Console().print(table)


@lens_app.command("stats")
def show_statistics(
    pipeline: Optional[str] = typer.Option(None, help="Filter by pipeline name"),
    hours: int = typer.Option(24, help="Time range in hours from now"),
) -> None:
    """Show aggregated span statistics."""
    backend = load_backend_from_config()
    try:
        import time

        end_time = time.time()
        start_time = end_time - (hours * 3600)
        time_range = (start_time, end_time)

        stats = asyncio.run(
            backend.get_span_statistics(pipeline_name=pipeline, time_range=time_range)
        )
    except NotImplementedError:
        typer.echo("Backend does not support span statistics", err=True)
        raise typer.Exit(1)

    console = Console()

    # Overall statistics
    console.print(f"[bold]Span Statistics (last {hours} hours):[/bold]")
    console.print(f"Total spans: {stats['total_spans']}")

    # Status breakdown
    if stats["by_status"]:
        console.print("\n[bold]By Status:[/bold]")
        status_table = Table("Status", "Count")
        for status, count in stats["by_status"].items():
            status_table.add_row(status, str(count))
        console.print(status_table)

    # Name breakdown
    if stats["by_name"]:
        console.print("\n[bold]By Name:[/bold]")
        name_table = Table("Name", "Count")
        for name, count in stats["by_name"].items():
            name_table.add_row(name, str(count))
        console.print(name_table)

    # Duration statistics
    if stats["avg_duration_by_name"]:
        console.print("\n[bold]Average Duration by Name:[/bold]")
        duration_table = Table("Name", "Average Duration", "Count")
        for name, data in stats["avg_duration_by_name"].items():
            if data["count"] > 0:
                duration_table.add_row(name, f"{data['average']:.2f}s", str(data["count"]))
        console.print(duration_table)
