from __future__ import annotations
import typer
import asyncio
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json
from .config import load_backend_from_config
import os as __os
from typing import Optional, Any


def _find_run_by_partial_id(backend: Any, partial_id: str, timeout: float = 5.0) -> Optional[str]:
    """Find a run by partial run_id match. Returns full run_id or None."""
    try:

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

        result: Optional[str] = asyncio.run(asyncio.wait_for(_search(), timeout=timeout))
        return result
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
    """Show detailed information about a run, with optional step input/output/error."""
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

    # Fast path in CI/tests for SQLite: avoid event loop and rich rendering
    # Fast-mode heuristics rely only on env to avoid expensive settings init
    _fast_mode = (
        bool(__os.getenv("PYTEST_CURRENT_TEST"))
        or (__os.getenv("CI", "").lower() in ("true", "1"))
        or (__os.getenv("FLUJO_TEST_MODE", "").strip() in ("1", "true", "True"))
    )

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
                db_path = getattr(backend, "db_path")
                with _sqlite3.connect(db_path) as _conn:
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

            details, steps = asyncio.run(asyncio.wait_for(_fetch(), timeout=timeout))
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
            exec_time_ms = details['execution_time_ms']
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
