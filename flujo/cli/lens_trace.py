from __future__ import annotations
import typer
from typing import Any, Optional
import datetime
import json as _json
from .config import load_backend_from_config
from flujo.type_definitions.common import JSONObject
from flujo.utils.async_bridge import run_sync


def _format_node_label(node: JSONObject) -> str:
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


def _render_trace_tree(
    node: JSONObject, parent: Optional[Any] = None, *, preview_len: int = 200
) -> Any:
    try:
        from rich.tree import Tree

        label = _format_node_label(node)
        tree = Tree(label) if parent is None else parent.add(label)
        # Render notable events (e.g., agent.prompt) under the span
        events = node.get("events") or []
        try:
            for ev in events:
                name = str(ev.get("name"))
                if name == "agent.prompt":
                    attrs = ev.get("attributes", {}) or {}
                    preview = str(attrs.get("rendered_history", ""))
                    if preview_len is not None and preview_len >= 0 and len(preview) > preview_len:
                        preview = preview[:200] + "..."
                    tree.add(f"[dim]event[/dim] [cyan]{name}[/cyan]: {preview}")
        except Exception:
            pass
        for child in node.get("children", []):
            _render_trace_tree(child, tree, preview_len=preview_len)
        return tree
    except ModuleNotFoundError:
        # Fallback: compose a plain text tree
        label = _format_node_label(node)
        lines = [label]
        for child in node.get("children", []):
            child_repr = _render_trace_tree(child, None, preview_len=preview_len)
            for ln in str(child_repr).splitlines():
                lines.append("  " + ln)
        return "\n".join(lines)


def _convert_to_timestamp(val: Any) -> Optional[float]:
    """Convert a value to a timestamp, handling exceptions."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def trace_command(run_id: str, *, prompt_preview_len: int = 200) -> None:
    """Show the hierarchical execution trace for a run as a tree, with a summary."""
    backend = load_backend_from_config()
    try:
        trace = run_sync(backend.get_trace(run_id))
        run_details = None
        if hasattr(backend, "get_run_details"):
            try:
                run_details = run_sync(backend.get_run_details(run_id))
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

    # helper defs removed; using top-level versions

    def _print_trace_summary(trace: JSONObject, run_details: Optional[JSONObject] = None) -> None:
        console: Any | None = None
        panel_cls: Any | None = None
        text_cls: Any | None = None
        try:
            from rich.console import Console
            from rich.panel import Panel as RichPanel
            from rich.text import Text as RichText

            console = Console()
            panel_cls = RichPanel
            text_cls = RichText
        except ModuleNotFoundError:
            pass
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
        if console is not None and panel_cls is not None and text_cls is not None:
            summary = text_cls()
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
            console.print(
                panel_cls(summary, title="[bold cyan]Run Summary[/bold cyan]", expand=False)
            )
        else:
            typer.echo(f"Run ID: {run_id}")
            if pipeline:
                typer.echo(f"Pipeline: {pipeline}")
            typer.echo(f"Status: {status}")
            if start:
                typer.echo(f"Start: {fmt_time(start)}")
            if end:
                typer.echo(f"End: {fmt_time(end)}")
            if duration:
                typer.echo(f"Duration: {duration}")
            if steps:
                typer.echo(f"Steps: {steps}")

    _print_trace_summary(trace, run_details)
    tree = _render_trace_tree(trace, preview_len=prompt_preview_len)
    try:
        from rich.console import Console

        Console().print(tree)
    except ModuleNotFoundError:
        typer.echo(str(tree))


def trace_from_file(file_path: str, *, prompt_preview_len: int = 200) -> None:
    """Render a saved debug JSON (from --debug-export) as a rich trace tree.

    Accepts either a full export payload (with a 'trace_tree' key) or a bare
    trace root object.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            payload = _json.load(fh)
    except Exception as e:
        from .helpers import print_rich_or_typer

        print_rich_or_typer(f"[red]Failed to read file:[/red] {e}", stderr=True)
        raise typer.Exit(1) from e

    # Detect shape
    if isinstance(payload, dict) and "trace_tree" in payload:
        trace = payload.get("trace_tree")
        run_details = {
            "run_id": payload.get("run_id"),
            "pipeline_name": payload.get("pipeline_name"),
            "status": None,
            "created_at": payload.get("exported_at"),
            "end_time": None,
            "total_steps": None,
        }
    else:
        trace = payload
        run_details = None

    if not isinstance(trace, dict):
        from .helpers import print_rich_or_typer

        print_rich_or_typer("[red]Invalid trace payload: expected a JSON object[/red]", stderr=True)
        raise typer.Exit(1)

    # Print a lightweight summary when available
    try:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            console = Console()
            console.rule("Trace (from file)")
            if run_details is not None:
                rid = run_details.get("run_id") or "-"
                pipe = run_details.get("pipeline_name") or "-"
                created = run_details.get("created_at") or "-"
                summary = Text()
                summary.append(f"Run ID: {rid}\n")
                summary.append(f"Pipeline: {pipe}\n")
                summary.append(f"Exported At: {created}\n")
                console.print(Panel(summary, title="Summary", expand=False))
            tree = _render_trace_tree(trace, preview_len=prompt_preview_len)
            console.print(tree)
        except ModuleNotFoundError:
            # Fallback to plain text
            if run_details is not None:
                rid = run_details.get("run_id") or "-"
                pipe = run_details.get("pipeline_name") or "-"
                created = run_details.get("created_at") or "-"
                typer.echo(f"Run ID: {rid}")
                typer.echo(f"Pipeline: {pipe}")
                typer.echo(f"Exported At: {created}")
            tree = _render_trace_tree(trace, preview_len=prompt_preview_len)
            typer.echo(str(tree))
    except Exception as e:
        from .helpers import print_rich_or_typer

        print_rich_or_typer(f"[red]Failed to render trace:[/red] {e}", stderr=True)
        raise typer.Exit(1) from e
