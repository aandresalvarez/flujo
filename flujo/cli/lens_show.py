from __future__ import annotations
import typer
import asyncio
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json
from .config import load_backend_from_config
from typing import Optional, Any
from flujo.utils.async_bridge import run_sync
from flujo.utils.config import get_settings


def _find_run_by_partial_id(backend: Any, partial_id: str, timeout: float = 5.0) -> Optional[str]:
    """
    Find a run by partial run_id match with fuzzy search support.

    This function enables users to specify only the first 8-12 characters of a run_id
    instead of the full 32+ character ID. It first attempts an exact match, then falls
    back to prefix matching across recent runs.

    Args:
        backend: The state backend instance (SQLite, memory, etc.)
        partial_id: Partial or complete run_id to search for
        timeout: Maximum time in seconds to wait for search (default: 5.0s)

    Returns:
        The full run_id if a unique match is found, None otherwise

    Raises:
        ValueError: If multiple runs match the partial_id (ambiguous)
        asyncio.TimeoutError: If the search exceeds the timeout (caught internally)

    Examples:
        >>> backend = load_backend_from_config()
        >>> full_id = _find_run_by_partial_id(backend, "abc123")
        >>> # Returns "abc123def456..." if unique match found
    """
    try:
        # Fast path for SQLite: avoid async bridge + event loop overhead during lookups.
        try:
            from flujo.state.backends.sqlite import SQLiteBackend as _SQLiteBackend
            import sqlite3 as _sqlite3

            if isinstance(backend, _SQLiteBackend) and hasattr(backend, "db_path"):
                db_path = backend.db_path
                with _sqlite3.connect(str(db_path), timeout=timeout) as _conn:
                    _conn.row_factory = _sqlite3.Row

                    # Try exact match first (indexed by PRIMARY KEY).
                    cur = _conn.execute(
                        "SELECT run_id FROM runs WHERE run_id = ? LIMIT 1", (partial_id,)
                    )
                    row = cur.fetchone()
                    cur.close()
                    if row is not None:
                        return str(row["run_id"])

                    # Prefix match via range scan: [prefix, prefix + U+FFFF)
                    upper_bound = f"{partial_id}\uffff"
                    cur = _conn.execute(
                        "SELECT run_id FROM runs WHERE run_id >= ? AND run_id < ? LIMIT 6",
                        (partial_id, upper_bound),
                    )
                    rows = cur.fetchall()
                    cur.close()
                    matches = [str(r["run_id"]) for r in rows]
                    if len(matches) == 1:
                        return matches[0]
                    if len(matches) > 1:
                        raise ValueError(f"Ambiguous run_id '{partial_id}'. Matches: {matches[:5]}")
                    return None
        except ValueError:
            raise
        except Exception:
            pass  # Intentional: fall back to async path on any DB error

        async def _search() -> Optional[str]:
            try:
                # Try exact match first
                details = await backend.get_run_details(partial_id)
                if details:
                    return partial_id
            except Exception:
                pass

            # Try partial match
            if hasattr(backend, "list_runs"):
                runs = await backend.list_runs(limit=100)
            else:
                runs = await backend.list_workflows(limit=100)

            matches = [str(r["run_id"]) for r in runs if str(r["run_id"]).startswith(partial_id)]
            if len(matches) == 1:
                return str(matches[0])
            elif len(matches) > 1:
                raise ValueError(f"Ambiguous run_id '{partial_id}'. Matches: {matches[:5]}")
            return None

        result: Optional[str] = run_sync(asyncio.wait_for(_search(), timeout=timeout))
        return result
    except ValueError:
        raise
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


def show_run(
    run_id: str,
    show_output: bool = False,
    show_input: bool = False,
    show_error: bool = False,
    verbose: bool = False,
    json_output: bool = False,
    show_final_output: bool = False,
    timeout: float = 10.0,
) -> None:
    """
    Display detailed information about a workflow run with rich formatting.

    This function fetches run details and steps from the configured state backend,
    supporting both full and partial run_id matches. It provides multiple output
    modes (human-readable tables, JSON) and detailed step-by-step inspection.

    Args:
        run_id: Full or partial run_id to display (supports 8+ char prefix matching)
        show_output: Include step outputs in the display (default: False)
        show_input: Include step inputs in the display (default: False)
        show_error: Include step errors in the display (default: False)
        verbose: Show all step details (input, output, error) (default: False)
        json_output: Output as structured JSON instead of rich formatting (default: False)
        show_final_output: Display only the final pipeline output (default: False)
        timeout: Maximum time in seconds to wait for backend queries (default: 10.0s)

    Raises:
        typer.Exit(1): If run not found, timeout occurs, or backend errors

    Examples:
        >>> show_run("abc123def456", verbose=True)
        # Displays full run details with all step I/O

        >>> show_run("abc123", json_output=True)
        # Outputs JSON for the run matching partial ID "abc123"

    Notes:
        - Supports partial run_id matching (minimum 8 characters recommended)
        - Fast mode automatically enabled in CI/test environments
        - Use FLUJO_LENS_TIMEOUT env var to override default timeout
        - JSON mode is ideal for CI/CD automation and scripting
    """
    backend = load_backend_from_config()

    # Try partial run_id matching
    if len(run_id) < 30:  # run_ids are typically 32+ chars
        try:
            full_run_id: Optional[str] = _find_run_by_partial_id(backend, run_id, timeout=2.0)
            if full_run_id:
                if not json_output:
                    Console().print(f"[dim]Matched partial ID to: {full_run_id}[/dim]")
                run_id = full_run_id
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        except Exception:
            pass  # Continue with original run_id

    # Fast path in test environments for SQLite: avoid event loop and rich rendering
    _fast_mode = bool(get_settings().test_mode)

    # If detailed output was requested, disable fast mode to fetch full payloads
    if verbose or show_input or show_output or show_error or json_output or show_final_output:
        _fast_mode = False

    details = None
    steps = None
    if _fast_mode:
        try:
            from flujo.state.backends.sqlite import SQLiteBackend as _SB
            import sqlite3 as _sqlite3

            if isinstance(backend, _SB) and hasattr(backend, "db_path"):
                db_path = backend.db_path
                with _sqlite3.connect(str(db_path)) as _conn:
                    _conn.row_factory = _sqlite3.Row
                    cur = _conn.execute(
                        (
                            "SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, "
                            "execution_time_ms, memory_usage_mb, total_steps, error_message FROM runs WHERE run_id = ?"
                        ),
                        (run_id,),
                    )
                    row = cur.fetchone()
                    cur.close()
                    if row is not None:
                        details = {
                            "run_id": row["run_id"],
                            "pipeline_name": row["pipeline_name"],
                            "pipeline_version": row["pipeline_version"],
                            "status": row["status"],
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                            "execution_time_ms": row["execution_time_ms"],
                            "memory_usage_mb": row["memory_usage_mb"],
                            "total_steps": row["total_steps"],
                            "error_message": row["error_message"],
                        }
                    # Only grab minimal step fields for speed when not verbose
                    cur2 = _conn.execute(
                        (
                            "SELECT step_index, step_name, status FROM steps WHERE run_id = ? ORDER BY step_index"
                        ),
                        (run_id,),
                    )
                    rows = cur2.fetchall()
                    cur2.close()
                    steps = [
                        {
                            "step_index": r["step_index"],
                            "step_name": r["step_name"],
                            "status": r["status"],
                        }
                        for r in rows
                    ]
        except Exception:
            details = None
            steps = None

    if details is None or steps is None:
        try:

            async def _fetch() -> tuple[dict[str, object] | None, list[dict[str, object]]]:
                d_task = asyncio.create_task(backend.get_run_details(run_id))
                s_task = asyncio.create_task(backend.list_run_steps(run_id))
                return await d_task, await s_task

            details, steps = run_sync(asyncio.wait_for(_fetch(), timeout=timeout))
        except asyncio.TimeoutError:
            typer.echo(
                f"Timeout ({timeout}s) while fetching run details\n"
                "Suggestions:\n"
                "  • Try increasing timeout with FLUJO_LENS_TIMEOUT env var\n"
                "  • Check if the database is locked by another process\n"
                f"  • Use 'flujo lens list' to verify run exists: {run_id}",
                err=True,
            )
            raise typer.Exit(1)
        except NotImplementedError:
            typer.echo("Backend does not support run inspection", err=True)
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(
                f"Error accessing backend: {e}\n"
                f"Run ID: {run_id}\n"
                "Suggestions:\n"
                "  • Verify the run_id exists with 'flujo lens list'\n"
                "  • Check database permissions\n"
                "  • Try with a different backend (memory:// for testing)",
                err=True,
            )
            raise typer.Exit(1)

    if details is None:
        typer.echo(
            f"Run not found: {run_id}\n"
            "Suggestions:\n"
            "  • Check run_id with 'flujo lens list'\n"
            "  • Try partial ID match (e.g., first 8-12 characters)\n"
            "  • Verify correct state_uri in flujo.toml",
            err=True,
        )
        raise typer.Exit(1)
    assert details is not None

    # JSON output mode
    if json_output:
        output = {
            "run_id": run_id,
            "details": details,
            "steps": steps,
        }
        print(json.dumps(output, indent=2, default=str))
        return

    if _fast_mode:
        # Minimal, tab-delimited for speed and easy parsing
        print(f"Run\t{run_id}\t{details['status']}")
        if steps:
            for s in steps:
                print(f"{s.get('step_index')}\t{s.get('step_name', '-')}\t{s.get('status', '-')}")
    else:
        console = Console()

        # Show run summary
        summary = Text()
        summary.append("Run ID: ", style="bold")
        summary.append(f"{run_id}\n")
        summary.append("Pipeline: ", style="bold")
        summary.append(f"{details.get('pipeline_name', '-')}\n")
        summary.append("Status: ", style="bold")
        status = str(details.get("status", "unknown"))
        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(
            status.lower(), "white"
        )
        summary.append(f"{status}\n", style=status_color)

        if details.get("execution_time_ms"):
            summary.append("Duration: ", style="bold")
            exec_time_ms = details["execution_time_ms"]
            if isinstance(exec_time_ms, (int, float)):
                summary.append(f"{exec_time_ms / 1000:.2f}s\n")
            else:
                summary.append(f"{exec_time_ms}\n")
        if details.get("total_steps"):
            summary.append("Total Steps: ", style="bold")
            summary.append(f"{details['total_steps']}\n")
        if details.get("created_at"):
            summary.append("Created: ", style="bold")
            summary.append(f"{details['created_at']}\n")

        console.print(Panel(summary, title="[bold cyan]Run Summary[/bold cyan]", expand=False))

        # Show steps table
        if steps:
            table = Table("Index", "Step Name", "Status", "Time (ms)", title="Steps")
            for s in steps:
                exec_time = s.get("execution_time_ms", "-")
                table.add_row(
                    str(s.get("step_index", "-")),
                    str(s.get("step_name", "-")),
                    str(s.get("status", "-")),
                    f"{exec_time:.0f}" if isinstance(exec_time, (int, float)) else str(exec_time),
                )
            console.print(table)

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

    if (show_input or show_output or show_error) and not _fast_mode and steps:
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

    # Show final output if requested
    if show_final_output and not _fast_mode and steps:
        console = Console()
        final_step = steps[-1] if steps else None
        if final_step and final_step.get("output") is not None:
            try:
                pretty = json.dumps(final_step["output"], indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                pretty = str(final_step["output"])
            console.print(
                Panel(
                    pretty,
                    title="[bold green]Final Output[/bold green]",
                    expand=False,
                )
            )
