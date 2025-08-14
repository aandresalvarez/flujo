from __future__ import annotations
import typer
import asyncio
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
import json
from .config import load_backend_from_config


def show_run(
    run_id: str,
    show_output: bool = False,
    show_input: bool = False,
    show_error: bool = False,
    verbose: bool = False,
) -> None:
    """Show detailed information about a run, with optional step input/output/error."""
    backend = load_backend_from_config()
    try:
        details = asyncio.run(backend.get_run_details(run_id))
        steps = asyncio.run(backend.list_run_steps(run_id))
    except NotImplementedError:
        typer.echo("Backend does not support run inspection", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error accessing backend: {e}", err=True)
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

    # If the run failed before any step executed, surface the error message for clarity
    if details.get("status") == "failed" and (not steps or len(steps) == 0):
        err_msg = details.get("error_message") or details.get("reason") or "Unknown error"
        Console().print(
            Panel(
                f"[bold red]Failure before first step[/bold red]\n{err_msg}",
                title="Run Error",
                expand=False,
            )
        )

    # Determine which fields to show
    show_input = show_input or verbose
    show_output = show_output or verbose
    show_error = show_error or verbose

    if show_input or show_output or show_error:
        console = Console()
        for s in steps:
            step_idx = s.get("step_index")
            step_name = s.get("step_name", "-")
            lines = []
            if show_input and s.get("input") is not None:
                try:
                    pretty = json.dumps(s["input"], indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    pretty = str(s["input"])
                lines.append(f"[bold]Input:[/bold]\n{pretty}")
            if show_output and s.get("output") is not None:
                try:
                    pretty = json.dumps(s["output"], indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    pretty = str(s["output"])
                lines.append(f"[bold]Output:[/bold]\n{pretty}")
            if show_error and s.get("error") is not None:
                try:
                    pretty = json.dumps(s["error"], indent=2, ensure_ascii=False)
                except (TypeError, ValueError):
                    pretty = str(s["error"])
                lines.append(f"[bold]Error:[/bold]\n{pretty}")
            if lines:
                panel_title = f"Step {step_idx}: {step_name}"
                console.print(Panel("\n\n".join(lines), title=panel_title, expand=False))
