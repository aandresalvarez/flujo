from __future__ import annotations
import typer
import asyncio
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from typing import Dict, Any, Optional
import datetime
from .config import load_backend_from_config


def _convert_to_timestamp(val: Any) -> Optional[float]:
    """Convert a value to a timestamp, handling exceptions."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def trace_command(run_id: str) -> None:
    """Show the hierarchical execution trace for a run as a tree, with a summary."""
    backend = load_backend_from_config()
    try:
        trace = asyncio.run(backend.get_trace(run_id))
        run_details = None
        if hasattr(backend, "get_run_details"):
            try:
                run_details = asyncio.run(backend.get_run_details(run_id))
            except Exception:
                run_details = None
    except NotImplementedError:
        typer.echo(
            f"The configured '{type(backend).__name__}' backend does not support trace inspection.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error accessing backend: {e}", err=True)
        raise typer.Exit(1)

    if not trace:
        typer.echo(f"No trace found for run_id: {run_id}", err=True)
        typer.echo("This could mean:", err=True)
        typer.echo("  - The run_id doesn't exist", err=True)
        typer.echo("  - The run completed without trace data", err=True)
        typer.echo("  - The backend doesn't support trace storage", err=True)
        raise typer.Exit(1)

    def _format_node_label(node: Dict[str, Any]) -> str:
        name = node.get("name", "(unknown)")
        status = node.get("status", "unknown")
        start = node.get("start_time")
        end = node.get("end_time")
        duration = None
        # Use robust timestamp conversion to handle string, int, float, or None
        start_timestamp = _convert_to_timestamp(start)
        end_timestamp = _convert_to_timestamp(end)
        if start_timestamp is not None and end_timestamp is not None:
            duration = end_timestamp - start_timestamp
        status_icon = "✅" if status == "completed" else ("❌" if status == "failed" else "⏳")
        label = f"{status_icon} [bold]{name}[/bold]"
        if duration is not None:
            label += f" [dim](duration: {duration:.2f}s)[/dim]"
        attrs = node.get("attributes", {})
        if attrs:
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)
            if attr_str:
                label += f" [dim]{attr_str}[/dim]"
        return label

    def _render_trace_tree(node: Dict[str, Any], parent: Optional[Tree] = None) -> Tree:
        label = _format_node_label(node)
        tree = Tree(label) if parent is None else parent.add(label)
        for child in node.get("children", []):
            _render_trace_tree(child, tree)
        return tree

    def _print_trace_summary(
        trace: Dict[str, Any], run_details: Optional[Dict[str, Any]] = None
    ) -> None:
        console = Console()
        run_id = run_details.get("run_id") if run_details else trace.get("run_id")
        pipeline = run_details.get("pipeline_name") if run_details else trace.get("name")
        status = run_details.get("status") if run_details else trace.get("status")
        start = run_details.get("created_at") if run_details else trace.get("start_time")
        end = run_details.get("end_time") if run_details else trace.get("end_time")
        steps = run_details.get("total_steps") if run_details else None

        def fmt_time(val: Any) -> str:
            if not val:
                return "-"
            try:
                if isinstance(val, (int, float)):
                    if float(val) < 0:
                        return "<invalid-timestamp>"
                    return datetime.datetime.fromtimestamp(float(val)).isoformat()
                return str(val)
            except (ValueError, TypeError):
                return str(val)

        duration = None
        start_ts = _convert_to_timestamp(start)
        end_ts = _convert_to_timestamp(end)
        if start_ts is not None and end_ts is not None:
            duration = f"{end_ts - start_ts:.2f}s"
        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(
            str(status).lower(), "white"
        )
        summary = Text()
        summary.append(f"Run ID: {run_id}\n", style="bold")
        if pipeline:
            summary.append(f"Pipeline: {pipeline}\n")
        summary.append("Status: ", style="bold")
        summary.append(f"{status}\n", style=status_color)
        if start:
            summary.append(f"Start: {fmt_time(start)}\n")
        if end:
            summary.append(f"End: {fmt_time(end)}\n")
        if duration:
            summary.append(f"Duration: {duration}\n")
        if steps:
            summary.append(f"Steps: {steps}\n")
        console.print(Panel(summary, title="[bold cyan]Run Summary[/bold cyan]", expand=False))

    _print_trace_summary(trace, run_details)
    tree = _render_trace_tree(trace)
    Console().print(tree)
