from __future__ import annotations

import typer
import asyncio
from typing import Dict, Any, Optional
from rich.table import Table
from rich.console import Console
from rich.tree import Tree

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


@lens_app.command("trace")
def show_trace(run_id: str) -> None:
    """Show the hierarchical execution trace for a run as a tree."""
    backend = load_backend_from_config()
    try:
        trace = asyncio.run(backend.get_trace(run_id))
    except NotImplementedError:
        typer.echo("Backend does not support trace inspection", err=True)
        raise typer.Exit(1)
    if not trace:
        typer.echo(f"No trace found for run_id: {run_id}", err=True)
        raise typer.Exit(1)

    def _render_trace_tree(node: Dict[str, Any], parent: Optional[Tree] = None) -> Tree:
        # Compose label: name, status, duration, attributes
        name = node.get("name", "(unknown)")
        status = node.get("status", "unknown")
        start = node.get("start_time")
        end = node.get("end_time")
        duration = None
        if start is not None and end is not None:
            try:
                duration = float(end) - float(start)
            except Exception:
                duration = None
        status_icon = "✅" if status == "completed" else ("❌" if status == "failed" else "⏳")
        label = f"{status_icon} [bold]{name}[/bold]"
        if duration is not None:
            label += f" [dim](duration: {duration:.2f}s)[/dim]"
        # Show key attributes
        attrs = node.get("attributes", {})
        if attrs:
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)
            if attr_str:
                label += f" [dim]{attr_str}[/dim]"
        tree = Tree(label) if parent is None else parent.add(label)
        for child in node.get("children", []):
            _render_trace_tree(child, tree)
        return tree

    tree = _render_trace_tree(trace)
    Console().print(tree)
