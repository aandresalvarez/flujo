import typer
from typing import Optional, Union, Any
import os as _os
import runpy
from rich.table import Table
from rich.console import Console
from .config import load_backend_from_config
from .helpers import find_project_root, load_pipeline_from_yaml_file
from .lens_show import show_run
from .lens_trace import trace_command, trace_from_file
from flujo.utils.async_bridge import run_sync
from flujo.utils.config import get_settings

lens_app = typer.Typer(
    rich_markup_mode="markdown",
    help=(
        "🔍 Inspect, debug, and trace past workflow runs.\n\n"
        "Commands:\n"
        "- `list`: show recent runs\n"
        "- `get <partial_run_id>`: find and show run by partial ID\n"
        "- `show <run_id>`: step-by-step details (supports partial IDs)\n"
        "- `trace <run_id>`: rich hierarchical trace tree\n"
        "- `spans <run_id>`: list individual spans with filtering\n"
        "- `stats`: aggregated span statistics\n"
        "- `from-file <path>`: render a saved debug export (from `--debug-export`)\n"
        "- `replay <run_id>`: re-run deterministically using recorded responses (when available)\n"
    ),
)


def register_lens_app(app: typer.Typer) -> None:
    """Register the lens sub-app with the main CLI app.

    Args:
        app: The main Typer app instance to register the lens app with.
    """
    app.add_typer(lens_app, name="lens")


@lens_app.command("list")
def list_runs(
    status: Union[str, None] = typer.Option(None),
    pipeline: Union[str, None] = typer.Option(None),
    limit: int = typer.Option(50, help="Maximum number of runs to display"),
) -> None:
    """List stored runs."""
    backend = load_backend_from_config()
    # Ultra-fast path for SQLite in CI/tests: use sqlite3 directly to avoid event loop overhead
    runs = None
    try:
        from flujo.state.backends.sqlite import SQLiteBackend as _SB

        if isinstance(backend, _SB) and hasattr(backend, "db_path"):
            import sqlite3 as _sqlite3

            db_path = getattr(backend, "db_path")
            with _sqlite3.connect(db_path) as _conn:
                _conn.row_factory = _sqlite3.Row
                params: list[object] = []
                query = "SELECT run_id, pipeline_name, status, created_at FROM runs "
                have_where = False
                if status:
                    query += "WHERE status = ? "
                    params.append(status)
                    have_where = True
                if pipeline:
                    query += ("AND " if have_where else "WHERE ") + "pipeline_name = ? "
                    params.append(pipeline)
                    have_where = True
                query += "ORDER BY created_at DESC LIMIT ?"
                params.append(int(limit))
                cur = _conn.execute(query, params)
                rows = cur.fetchall()
                runs = [
                    {
                        "run_id": r["run_id"],
                        "pipeline_name": r["pipeline_name"],
                        "status": r["status"],
                        "created_at": r["created_at"],
                    }
                    for r in rows
                ]
    except Exception:
        runs = None

    if runs is None:
        try:
            # Use the new structured API if available, fallback to legacy
            if hasattr(backend, "list_runs"):
                runs = run_sync(
                    backend.list_runs(status=status, pipeline_name=pipeline, limit=limit)
                )
            else:
                runs = run_sync(
                    backend.list_workflows(status=status, pipeline_id=pipeline, limit=limit)
                )
        except NotImplementedError:
            typer.echo("Backend does not support listing runs", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error accessing backend: {e}", err=True)
            raise typer.Exit(1)

    # Minimal, fast output for test environments to reduce rendering overhead.
    _fast_mode = bool(get_settings().test_mode)

    if _fast_mode:
        out_lines = []
        for r in runs:
            created_at = r.get("created_at") or r.get("start_time") or "-"
            out_lines.append(
                f"{r.get('run_id', '-')}\t{r.get('pipeline_name', '-')}\t{r.get('status', '-')}\t{created_at}"
            )
        print("\n".join(out_lines))
    else:
        table = Table("run_id", "pipeline", "status", "created_at")
        for r in runs:
            # Prefer created_at field exposed by backends; fall back gracefully
            created_at = r.get("created_at") or r.get("start_time") or "-"
            table.add_row(
                r.get("run_id", "-"),
                r.get("pipeline_name", "-"),
                r.get("status", "-"),
                str(created_at),
            )
        Console().print(table)


@lens_app.command("get")
def get_by_partial_id(
    partial_id: str,
    show_output: bool = typer.Option(False, "--show-output", help="Show step outputs."),
    show_input: bool = typer.Option(False, "--show-input", help="Show step inputs."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show input, output, and error for each step."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
    show_final_output: bool = typer.Option(
        False, "--final-output", help="Show the final pipeline output."
    ),
) -> None:
    """
    Find and display runs by fuzzy search on partial run_id.

    This command enables quick lookup of runs using partial IDs (substring matching).
    If multiple matches are found, displays a disambiguation table. If a unique match
    is found, automatically shows the run details.

    Args:
        partial_id: Partial run_id to search for (matches anywhere in run_id)
        show_output: Include step outputs in the display
        show_input: Include step inputs in the display
        verbose: Show all step details (input, output, error)
        json_output: Output as structured JSON for automation
        show_final_output: Display only the final pipeline output

    Examples:
        flujo lens get abc123      # Match runs containing "abc123"
        flujo lens get run_       # Match all runs starting with "run_"
        flujo lens get abc --json  # JSON output for matched run

    Notes:
        - Uses substring matching (not just prefix)
        - Shows disambiguation table if multiple matches found
        - Automatically displays full run details for unique matches
    """
    backend = load_backend_from_config()

    # Async function to search for runs (no nested event loop)
    async def _search_runs() -> list[dict[str, Any]]:
        if hasattr(backend, "list_runs"):
            result = await backend.list_runs(limit=100)
        else:
            result = await backend.list_workflows(limit=100)
        return list(result) if result else []

    # Run async search first, then handle results synchronously
    try:
        runs = run_sync(_search_runs())
    except Exception as e:
        typer.echo(f"Error searching for runs: {e}", err=True)
        raise typer.Exit(1)

    # Find matches (synchronous)
    matches = [r["run_id"] for r in runs if partial_id in r["run_id"]]

    if len(matches) == 0:
        typer.echo(
            f"No runs found matching: {partial_id}\n"
            "Suggestions:\n"
            "  • Use 'flujo lens list' to see all runs\n"
            "  • Try a different substring\n"
            "  • Check if the run exists in the configured state backend",
            err=True,
        )
        raise typer.Exit(1)
    elif len(matches) > 1:
        console = Console()
        console.print(f"[yellow]Multiple matches found for '{partial_id}':[/yellow]")
        table = Table("Index", "Run ID", "Pipeline", "Status")
        for idx, match_id in enumerate(matches[:10], 1):
            run_info: dict[str, Any] = next((r for r in runs if r["run_id"] == match_id), {})
            table.add_row(
                str(idx),
                match_id[:16] + "..." if len(match_id) > 16 else match_id,
                run_info.get("pipeline_name", "-"),
                run_info.get("status", "-"),
            )
        console.print(table)
        if len(matches) > 10:
            console.print(f"[dim]... and {len(matches) - 10} more matches[/dim]")
        console.print("\n[yellow]Please provide a more specific run_id.[/yellow]")
        raise typer.Exit(1)
    else:
        # Exact match - show details (now safe to call show_run)
        full_run_id = matches[0]
        Console().print(f"[dim]Found: {full_run_id}[/dim]\n")
        show_run(
            full_run_id,
            show_output=show_output,
            show_input=show_input,
            verbose=verbose,
            json_output=json_output,
            show_final_output=show_final_output,
        )


@lens_app.command("show")
def show_command(
    run_id: str,
    show_output: bool = typer.Option(False, "--show-output", help="Show step outputs."),
    show_input: bool = typer.Option(False, "--show-input", help="Show step inputs."),
    show_error: bool = typer.Option(False, "--show-error", help="Show step errors."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show input, output, and error for each step."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
    show_final_output: bool = typer.Option(
        False, "--final-output", help="Show the final pipeline output."
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="Timeout in seconds for fetching data."),
) -> None:
    """
    Display comprehensive details for a specific workflow run.

    This command fetches and displays run metadata, step-by-step execution details,
    and optional I/O information. Supports both full and partial run_id matching
    with automatic prefix resolution for convenience.

    Args:
        run_id: Full or partial run_id (minimum 8 characters for partial matching)
        show_output: Include step outputs in the display
        show_input: Include step inputs in the display
        show_error: Include step errors in the display
        verbose: Show all step details (equivalent to all show_* flags enabled)
        json_output: Output as structured JSON for CI/CD automation
        show_final_output: Display only the final pipeline output
        timeout: Maximum wait time for backend queries (overridable via FLUJO_LENS_TIMEOUT env)

    Examples:
        flujo lens show run_abc123          # Show basic run info
        flujo lens show abc123 --verbose    # Use partial ID with full details
        flujo lens show abc123 --json       # JSON output for scripting
        flujo lens show abc123 --final-output  # Quick result check
        FLUJO_LENS_TIMEOUT=30 flujo lens show abc123  # Custom timeout

    Environment Variables:
        FLUJO_LENS_TIMEOUT: Override default timeout (seconds)

    Notes:
        - Partial IDs auto-resolve if unique (minimum 8 chars recommended)
        - Fast mode enabled automatically in CI/test environments
        - JSON mode ideal for automation and monitoring dashboards
    """
    # Allow timeout override via env var
    import os

    timeout_env = os.getenv("FLUJO_LENS_TIMEOUT")
    if timeout_env:
        try:
            timeout = float(timeout_env)
        except ValueError:
            pass

    show_run(
        run_id,
        show_output=show_output,
        show_input=show_input,
        show_error=show_error,
        verbose=verbose,
        json_output=json_output,
        show_final_output=show_final_output,
        timeout=timeout,
    )


@lens_app.command("trace")
def trace_command_cli(
    run_id: str,
    prompt_preview_len: int = typer.Option(
        200,
        "--prompt-preview-len",
        help="Max characters to show for agent.prompt previews (-1 for full).",
    ),
) -> None:
    trace_command(run_id, prompt_preview_len=prompt_preview_len)


@lens_app.command("from-file")
def trace_from_file_cli(
    path: str,
    prompt_preview_len: int = typer.Option(
        200,
        "--prompt-preview-len",
        help="Max characters to show for prompt/response previews (-1 for full).",
    ),
) -> None:
    """Render a saved debug JSON (from --debug-export) as a rich trace tree."""
    trace_from_file(path, prompt_preview_len=prompt_preview_len)


@lens_app.command("replay")
def replay_command(
    run_id: str,
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Path to the pipeline file (.py or pipeline.yaml)"
    ),
    object_name: str = typer.Option(
        "pipeline", "--object", "-o", help="Name of the pipeline variable in a Python file"
    ),
    state_uri: Optional[str] = typer.Option(
        None,
        "--state-uri",
        help="Override FLUJO_STATE_URI for this replay (e.g., sqlite:////abs/path/ops.db)",
    ),
    json_output: bool = typer.Option(
        False, "--json", "--json-output", help="Output raw JSON instead of formatted result"
    ),
) -> None:
    """Replay a prior run deterministically using recorded trace and responses."""
    # Allow inline override of state URI for one-off replays
    if state_uri:
        _os.environ["FLUJO_STATE_URI"] = state_uri

    backend = load_backend_from_config()

    try:
        from .helpers import (
            load_pipeline_from_file,
            create_flujo_runner,
            display_pipeline_results,
        )
        from flujo.application.runner import Flujo

        runner: Flujo[Any, Any, Any]

        # Case 1: Load explicitly from file/object
        if file:
            try:
                if file.endswith((".yaml", ".yml")):
                    pipeline_obj = load_pipeline_from_yaml_file(file)
                else:
                    pipeline_obj, _ = load_pipeline_from_file(file, object_name)
            except SystemExit:
                # Enhance error message with guidance
                from .helpers import print_rich_or_typer

                print_rich_or_typer(
                    (
                        f"[red]Could not load pipeline object '{object_name}' from file '{file}'.\n"
                        "Ensure the file exists and exports the pipeline variable (or pass --object).[/red]"
                    ),
                    stderr=True,
                )
                raise
            runner = create_flujo_runner(
                pipeline=pipeline_obj,
                context_model_class=None,
                initial_context_data=None,
            )
        else:
            # Case 2: Attempt to infer project pipeline.yaml
            try:
                root = find_project_root()
                pyaml = (root / "pipeline.yaml").resolve()
                if not pyaml.exists():
                    raise FileNotFoundError
                pipeline_obj = load_pipeline_from_yaml_file(str(pyaml))
                runner = create_flujo_runner(
                    pipeline=pipeline_obj,
                    context_model_class=None,
                    initial_context_data=None,
                )
            except Exception:
                # Fallback: infer from run metadata and optional local registry
                details = run_sync(backend.get_run_details(run_id))
                if not details:
                    from .helpers import print_rich_or_typer

                    print_rich_or_typer(
                        f"[red]Run not found: {run_id}. Cannot infer pipeline without --file.[/red]",
                        stderr=True,
                    )
                    raise typer.Exit(1)

                pipeline_name = details.get("pipeline_name")
                pipeline_version = details.get("pipeline_version") or "latest"
                if not pipeline_name:
                    from .helpers import print_rich_or_typer

                    print_rich_or_typer(
                        "[red]Pipeline name unavailable in run metadata. Provide --file to load the pipeline.[/red]",
                        stderr=True,
                    )
                    raise typer.Exit(1)

                # Discover a local registry
                registry = None
                try:
                    import os as __os

                    if __os.path.exists("registry.py"):
                        ns = runpy.run_path("registry.py")
                        registry = ns.get("registry")
                except Exception:
                    registry = None

                if registry is None:
                    from .helpers import print_rich_or_typer

                    print_rich_or_typer(
                        (
                            "[red]No --file provided and no local registry found (expected 'registry.py' with a 'registry' object).\n"
                            f"Run metadata indicates pipeline '{pipeline_name}' (version '{pipeline_version}').\n"
                            "Please provide --file pointing to the pipeline definition or add a registry for inference.[/red]"
                        ),
                        stderr=True,
                    )
                    raise typer.Exit(1)

                runner = Flujo(
                    pipeline=None,
                    registry=registry,
                    pipeline_name=pipeline_name,
                    pipeline_version=pipeline_version,
                )
        # Attach operations backend so replay can load trace and steps from the configured store
        try:
            runner.state_backend = backend
        except Exception:
            pass

        async def _run() -> Any:
            return await runner.replay_from_trace(run_id)

        result = run_sync(_run())

        if json_output:
            from flujo.utils.serialization import _robust_serialize_internal
            import json as _json

            try:
                serialized = _robust_serialize_internal(result)
            except Exception:
                serialized = f"<unserializable: {type(result).__name__}>"
            typer.echo(_json.dumps(serialized, indent=2))
        else:
            display_pipeline_results(result, run_id, json_output)

    except Exception as e:
        try:
            import os

            os.makedirs("output", exist_ok=True)
            with open("output/last_run_error.txt", "w") as f:
                f.write(repr(e))
        except Exception:
            pass
        from .helpers import print_rich_or_typer

        print_rich_or_typer(f"[red]Replay failed: {e}", stderr=True)
        raise typer.Exit(1) from e


@lens_app.command("spans")
def list_spans(
    run_id: str,
    status: Optional[str] = typer.Option(None, help="Filter by span status"),
    name: Optional[str] = typer.Option(None, help="Filter by span name"),
) -> None:
    """List individual spans for a run with optional filtering."""
    backend = load_backend_from_config()
    try:
        spans = run_sync(backend.get_spans(run_id, status=status, name=name))
    except NotImplementedError:
        typer.echo("Backend does not support span-level querying", err=True)
        return
    except Exception as e:
        typer.echo(f"Error accessing backend: {e}", err=True)
        return

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
            except (ValueError, TypeError):
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

        stats = run_sync(backend.get_span_statistics(pipeline_name=pipeline, time_range=time_range))
    except NotImplementedError:
        typer.echo("Backend does not support span statistics", err=True)
        return
    except Exception as e:
        typer.echo(f"Error accessing backend: {e}", err=True)
        return

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


@lens_app.command("evals")
def list_evaluations(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter by run_id"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of evaluations to show"),
) -> None:
    """List shadow evaluation scores."""
    backend = load_backend_from_config()
    try:
        evals = run_sync(backend.list_evaluations(limit=limit, run_id=run_id))
    except NotImplementedError:
        typer.echo("Backend does not support listing evaluations", err=True)
        return
    except Exception as e:
        typer.echo(f"Error accessing backend: {e}", err=True)
        return

    if not evals:
        typer.echo("No evaluations found.")
        return

    table = Table("run_id", "step", "score", "feedback", "created_at")
    for item in evals:
        table.add_row(
            str(item.get("run_id", "")),
            str(item.get("step_name", "")),
            f"{item.get('score', '')}",
            (item.get("feedback") or "")[:60],
            str(item.get("created_at", "")),
        )

    Console().print(table)
