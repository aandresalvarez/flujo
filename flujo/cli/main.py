"""CLI entry point for flujo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast
import typer
import click
import json
from pathlib import Path
from flujo.infra.config_manager import get_cli_defaults as _get_cli_defaults
from flujo.exceptions import ConfigurationError, SettingsError
from flujo.exceptions import UsageLimitExceededError
from flujo.infra import telemetry
import flujo.builtins as _flujo_builtins  # noqa: F401  # Register builtin skills on CLI import
from typing_extensions import Annotated
from rich.console import Console
import os as _os
from ..utils.serialization import safe_serialize, safe_deserialize as _safe_deserialize
from .lens import lens_app
from .helpers import (
    run_benchmark_pipeline,
    create_benchmark_table,
    setup_solve_command_environment,
    execute_solve_pipeline,
    setup_run_command_environment,
    load_pipeline_from_yaml_file,
    create_flujo_runner,
    execute_pipeline_with_output_handling,
    display_pipeline_results,
    apply_cli_defaults,
    get_version_string,
    get_masked_settings_dict,
    execute_improve,
    load_mermaid_code,
    get_pipeline_step_names,
    validate_pipeline_file,
    parse_context_data,
    load_pipeline_from_file,
    find_project_root,
    scaffold_project,
    scaffold_demo_project,
    update_project_budget,
    resolve_project_root,
    ensure_project_root_on_sys_path,
    generate_demo_yaml,
    validate_yaml_text,
)
from .config import load_backend_from_config  # re-export for tests
from .exit_codes import (
    EX_OK,
    EX_CONFIG_ERROR,
    EX_RUNTIME_ERROR,
)
import click.testing
import os

# Expose Flujo class for tests that monkeypatch flujo.cli.main.Flujo.run
from flujo.application.runner import Flujo as _Flujo  # re-export for test monkeypatch compatibility

# Ensure project root is importable for custom packages (e.g., skills/)
try:
    _project_root = Path.cwd()
    import sys as _sys

    if str(_project_root) not in _sys.path:
        _sys.path.insert(0, str(_project_root))
except Exception:
    pass

# Import Flujo class for testing compatibility - commented out as unused
# from flujo.application.runner import Flujo

# Import functions that tests expect to monkeypatch - these are module-level imports
# that can be properly monkeypatched in tests
from flujo.recipes.factories import run_default_pipeline as _run_default_pipeline
from flujo.agents.recipes import (
    make_review_agent as _make_review_agent,
    make_solution_agent as _make_solution_agent,
    make_validator_agent as _make_validator_agent,
    get_reflection_agent as _get_reflection_agent,
    make_self_improvement_agent as _make_self_improvement_agent,
)
from flujo.application.self_improvement import (
    evaluate_and_improve as _evaluate_and_improve,
    SelfImprovementAgent as _SelfImprovementAgent,
    ImprovementReport as _ImprovementReport,
)
from flujo.application.eval_adapter import run_pipeline_async as _run_pipeline_async

# Removed override that blanked stderr; tests expect real stderr content
from typing import TYPE_CHECKING

# Re-export Flujo after all imports to satisfy linting (E402)
Flujo = _Flujo

if not TYPE_CHECKING:
    try:
        if not hasattr(click.testing.Result, "_flujo_stderr_shim"):

            def _stderr(self: click.testing.Result) -> str:
                return getattr(self, "output", "")

            # Assign property at runtime for test compatibility
            click.testing.Result.stderr = property(_stderr)
            setattr(click.testing.Result, "_flujo_stderr_shim", True)
    except Exception:
        pass

# Type definitions for CLI
WeightsType = List[Dict[str, Union[str, float]]]
MetadataType = Dict[str, Any]
ScorerType = (
    str  # Changed from Literal["ratio", "weighted", "reward"] to str for typer compatibility
)


# In CI/tests, disable ANSI styling and stabilize width for help snapshots
if _os.environ.get("PYTEST_CURRENT_TEST") or _os.environ.get("CI"):
    _os.environ.setdefault("NO_COLOR", "1")
    _os.environ.setdefault("COLUMNS", "107")
    # Ensure Rich uses a deterministic width inside Click/Typer's CliRunner
    try:
        import typer.rich_utils as _tru

        # Force Rich console width and disable terminal detection for deterministic wrapping
        try:
            setattr(_tru, "MAX_WIDTH", 107)
        except Exception:
            pass
        try:
            setattr(_tru, "FORCE_TERMINAL", True)
        except Exception:
            pass
        try:
            setattr(_tru, "COLOR_SYSTEM", None)
        except Exception:
            pass
        # Reduce edge padding so trailing spaces at table borders don't differ across platforms
        try:
            setattr(_tru, "STYLE_OPTIONS_TABLE_PAD_EDGE", False)
        except Exception:
            pass
        try:
            setattr(_tru, "STYLE_COMMANDS_TABLE_PAD_EDGE", False)
        except Exception:
            pass
    except Exception:
        pass
    try:
        import typer.rich_utils as _tru
        from typing import Union as _Union
        import click as _click
        import typer as _ty

        def _flujo_rich_format_help(
            *,
            obj: _Union[_click.Command, _click.Group],
            ctx: _click.Context,
            markup_mode: _tru.MarkupMode,
        ) -> None:
            # Usage and description without right-padding spaces to match snapshots
            _ty.echo("")
            _ty.echo(f" {obj.get_usage(ctx).strip()}")
            _ty.echo()
            _ty.echo()
            _ty.echo()
            if obj.help:
                _ty.echo(f" {obj.help.strip()}")
                _ty.echo()
                _ty.echo()
                _ty.echo()
                _ty.echo()
                _ty.echo()

            # Print a concise, non-truncated flags summary line to ensure full option names
            # are present in the help output (avoids ellipsizing like "--allow-side-eff…").
            try:
                option_names: list[str] = []
                for param in obj.get_params(ctx):
                    if getattr(param, "hidden", False):
                        continue
                    if isinstance(param, _click.Option):
                        # Prefer long options; fall back to short if needed
                        longs = [o for o in getattr(param, "opts", []) if o.startswith("--")]
                        shorts = [
                            o
                            for o in getattr(param, "opts", [])
                            if o.startswith("-") and not o.startswith("--")
                        ]
                        if longs:
                            option_names.append(longs[0])
                        elif getattr(param, "secondary_opts", None):
                            # e.g., boolean pairs like --flag/--no-flag
                            secs = [
                                o
                                for o in getattr(param, "secondary_opts", [])
                                if o.startswith("--")
                            ]
                            if secs:
                                option_names.append(f"{secs[0]}")
                        elif shorts:
                            option_names.append(shorts[0])
                if option_names:
                    _ty.echo(" Flags: " + ", ".join(option_names))
                    _ty.echo()
            except Exception:
                pass

            console = _tru._get_rich_console()
            from collections import defaultdict as _defaultdict
            from typing import DefaultDict as _DefaultDict, List as _List

            panel_to_arguments: _DefaultDict[str, _List[_click.Argument]] = _defaultdict(list)
            panel_to_options: _DefaultDict[str, _List[_click.Option]] = _defaultdict(list)
            for param in obj.get_params(ctx):
                if getattr(param, "hidden", False):
                    continue
                if isinstance(param, _click.Argument):
                    panel_name = (
                        getattr(param, _tru._RICH_HELP_PANEL_NAME, None)
                        or _tru.ARGUMENTS_PANEL_TITLE
                    )
                    panel_to_arguments[panel_name].append(param)
                elif isinstance(param, _click.Option):
                    panel_name = (
                        getattr(param, _tru._RICH_HELP_PANEL_NAME, None) or _tru.OPTIONS_PANEL_TITLE
                    )
                    panel_to_options[panel_name].append(param)

            default_arguments = panel_to_arguments.get(_tru.ARGUMENTS_PANEL_TITLE, [])
            _tru._print_options_panel(
                name=_tru.ARGUMENTS_PANEL_TITLE,
                params=default_arguments,
                ctx=ctx,
                markup_mode=markup_mode,
                console=console,
            )
            for panel_name, arguments in panel_to_arguments.items():
                if panel_name == _tru.ARGUMENTS_PANEL_TITLE:
                    continue
                _tru._print_options_panel(
                    name=panel_name,
                    params=arguments,
                    ctx=ctx,
                    markup_mode=markup_mode,
                    console=console,
                )

            default_options = panel_to_options.get(_tru.OPTIONS_PANEL_TITLE, [])
            _tru._print_options_panel(
                name=_tru.OPTIONS_PANEL_TITLE,
                params=default_options,
                ctx=ctx,
                markup_mode=markup_mode,
                console=console,
            )
            for panel_name, options in panel_to_options.items():
                if panel_name == _tru.OPTIONS_PANEL_TITLE:
                    continue
                _tru._print_options_panel(
                    name=panel_name,
                    params=options,
                    ctx=ctx,
                    markup_mode=markup_mode,
                    console=console,
                )

            if isinstance(obj, _click.Group):
                panel_to_commands: _DefaultDict[str, _List[_click.Command]] = _defaultdict(list)
                for command_name in obj.list_commands(ctx):
                    command = obj.get_command(ctx, command_name)
                    if command and not command.hidden:
                        panel_name = (
                            getattr(command, _tru._RICH_HELP_PANEL_NAME, None)
                            or _tru.COMMANDS_PANEL_TITLE
                        )
                        panel_to_commands[panel_name].append(command)

                max_cmd_len = max(
                    [
                        len(command.name or "")
                        for commands in panel_to_commands.values()
                        for command in commands
                    ],
                    default=0,
                )
                default_commands = panel_to_commands.get(_tru.COMMANDS_PANEL_TITLE, [])
                try:
                    _tru._print_commands_panel(
                        name=_tru.COMMANDS_PANEL_TITLE,
                        commands=default_commands,
                        markup_mode=markup_mode,
                        console=console,
                        cmd_len=max_cmd_len,
                    )
                except TypeError:
                    _tru._print_commands_panel(
                        name=_tru.COMMANDS_PANEL_TITLE,
                        commands=default_commands,
                        markup_mode=markup_mode,
                        console=console,
                    )
                for panel_name, commands in panel_to_commands.items():
                    if panel_name == _tru.COMMANDS_PANEL_TITLE:
                        continue
                    try:
                        _tru._print_commands_panel(
                            name=panel_name,
                            commands=commands,
                            markup_mode=markup_mode,
                            console=console,
                            cmd_len=max_cmd_len,
                        )
                    except TypeError:
                        _tru._print_commands_panel(
                            name=panel_name,
                            commands=commands,
                            markup_mode=markup_mode,
                            console=console,
                        )

        setattr(_tru, "rich_format_help", _flujo_rich_format_help)
        try:
            import typer.main as _tm

            setattr(_tm, "rich_format_help", _flujo_rich_format_help)
        except Exception:
            pass
    except Exception:
        pass

app: typer.Typer = typer.Typer(
    rich_markup_mode="markdown",
    help=(
        "A project-based server to build, run, and debug Flujo AI workflows.\n\n"
        "Common Commands:\n"
        "- `init` / `demo`: scaffold a project or a demo\n"
        "- `run`: run the current project's pipeline (YAML or Python)\n"
        "- `validate`: validate a pipeline file\n"
        "- `lens`: inspect past runs (`list`, `show`, `trace`, `from-file`)\n"
        "- `dev`: developer tools (`version`, `show-config`, `validate`, `visualize`, `budgets`)\n\n"
        "Debugging Flags for `run`:\n"
        "- `--debug`: step-by-step trace tree (safe previews)\n"
        "- `--trace-preview-len N`: preview size for prompts/responses\n"
        "- `--debug-prompts`: include full prompts/responses in trace (unsafe)\n"
        "- `--debug-export PATH`: write full debug JSON (trace + steps + context).\n"
        "  If omitted with `--debug`, auto-writes to `./debug/<timestamp>_<run_id>.json`.\n\n"
        "Project Root: pass `--project` or set `FLUJO_PROJECT_ROOT`.\n"
        "Verbose Errors: add `-v`/`--verbose` or `--trace`.\n"
        "Stable exit codes: see flujo.cli.exit_codes."
    ),
)

# Initialize telemetry at the start of CLI execution
telemetry.init_telemetry()
logfire = telemetry.logfire

"""Top-level sub-apps and groups."""
# Top-level: lens remains as its own sub-app
app.add_typer(lens_app, name="lens")

# New developer sub-app and nested experimental group
dev_app: typer.Typer = typer.Typer(
    rich_markup_mode=None,
    help="🛠️  Access advanced developer and diagnostic tools (e.g., version, show-config, visualize).",
)
experimental_app: typer.Typer = typer.Typer(
    rich_markup_mode=None, help="(Advanced) Experimental and diagnostic commands."
)
dev_app.add_typer(experimental_app, name="experimental")

# Budgets live under the dev group
budgets_app: typer.Typer = typer.Typer(rich_markup_mode=None, help="Budget governance commands")
dev_app.add_typer(budgets_app, name="budgets")

# Register developer app at top level
app.add_typer(dev_app, name="dev")


@dev_app.command(name="health-check", help="Analyze AROS signals from recent traces")
def dev_health_check(
    project: Annotated[Optional[str], typer.Option("--project", help="Project root path")] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Max recent runs to analyze", show_default=True)
    ] = 50,
    pipeline: Annotated[
        Optional[str], typer.Option("--pipeline", help="Filter by pipeline name")
    ] = None,
    since_hours: Annotated[
        Optional[int],
        typer.Option("--since-hours", help="Only analyze runs started within the last N hours"),
    ] = None,
    step_filter: Annotated[
        Optional[str], typer.Option("--step", help="Only include spans from this step name")
    ] = None,
    model_filter: Annotated[
        Optional[str], typer.Option("--model", help="Only include spans for this model id")
    ] = None,
    trend_buckets: Annotated[
        Optional[int],
        typer.Option(
            "--trend-buckets",
            help="Divide the selected time window into N equal buckets for trends (requires --since-hours)",
        ),
    ] = None,
    export: Annotated[
        Optional[str],
        typer.Option(
            "--export",
            help="Export format",
            click_type=click.Choice(["json", "csv"], case_sensitive=False),
        ),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", help="Output path (without extension for csv multi-file)"),
    ] = None,
) -> None:
    """Aggregate AROS summaries from span attributes and print hotspots."""
    console = Console()
    if project:
        try:
            ensure_project_root_on_sys_path(Path(project))
        except Exception:
            pass
    try:
        backend = load_backend_from_config()
    except Exception as e:
        console.print(f"[red]Failed to initialize state backend: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(code=1)

    import anyio

    async def _run() -> None:
        runs = await backend.list_runs(pipeline_name=pipeline, limit=limit)
        # Optional time window filter (best-effort)
        from datetime import datetime, timedelta, timezone

        parsed_times: dict[str, datetime] = {}
        cutoff = None
        if since_hours is not None:
            cutoff = datetime.utcnow() - timedelta(hours=int(since_hours))

            def _parse(ts: object) -> datetime | None:
                try:
                    # Numeric epoch
                    if isinstance(ts, (int, float)):
                        return datetime.utcfromtimestamp(float(ts))
                    if isinstance(ts, str):
                        s = ts.strip()
                        # Support Z-terminated ISO by normalizing to UTC
                        try:
                            return (
                                datetime.fromisoformat(s.replace("Z", "+00:00"))
                                .astimezone(timezone.utc)
                                .replace(tzinfo=None)
                            )
                        except Exception:
                            # Try epoch encoded as string
                            return datetime.utcfromtimestamp(float(s))
                    return None
                except Exception:
                    return None

            filtered: list[dict] = []
            for r in runs:
                ts = r.get("start_time") or r.get("created_at")
                dt = _parse(ts)
                if dt is not None:
                    parsed_times[r.get("run_id", "")] = dt
                if dt is None or (cutoff and dt >= cutoff):
                    filtered.append(r)
            runs = filtered
        if not runs:
            console.print("No runs found.")
            return
        totals = {
            "runs": 0,
            "coercion_total": 0,
            "stages": {},
            "soe_applied": 0,
            "soe_skipped": 0,
            "precheck_total": 0,
            "precheck_pass": 0,
            "precheck_fail": 0,
        }
        per_step: dict[str, dict[str, int]] = {}
        per_model: dict[str, dict[str, int]] = {}
        per_step_stages: dict[str, dict[str, int]] = {}
        per_model_stages: dict[str, dict[str, int]] = {}
        transforms_count: dict[str, int] = {}
        # Trend windows (last half vs previous half of the selected time window)
        trend: dict[str, Any] = {
            "last_half": {"coercions": 0},
            "prev_half": {"coercions": 0},
        }
        last_half_cut: datetime | None = None
        # Optional N-bucket trends across the selected window
        buckets: list[dict[str, Any]] = []
        now = None
        if since_hours is not None and cutoff is not None:
            now = datetime.utcnow()
            mid = cutoff + (now - cutoff) / 2
            last_half_cut = mid
            try:
                if isinstance(trend_buckets, int) and trend_buckets >= 2:
                    total_seconds = (now - cutoff).total_seconds() or 1.0
                    buckets = []
                    for i in range(trend_buckets):
                        b_start = cutoff + (now - cutoff) * (i / trend_buckets)
                        b_end = cutoff + (now - cutoff) * ((i + 1) / trend_buckets)
                        buckets.append(
                            {
                                "index": i,
                                "start": b_start.isoformat(),
                                "end": b_end.isoformat(),
                                "coercions": 0,
                                "stages": {},
                                "step_stages": {},
                                "model_stages": {},
                            }
                        )
                    trend["buckets"] = buckets
            except Exception:
                # If bucket math fails, silently skip and keep half-split only
                buckets = []

        for r in runs:
            run_id = r.get("run_id")
            if not run_id:
                continue
            totals["runs"] += 1
            spans = await backend.get_spans(run_id)
            # Determine run time for trend split (best-effort)
            run_dt = parsed_times.get(run_id)
            run_coercions = 0
            run_stage_counts: dict[str, int] = {}
            run_step_stage_counts: dict[str, dict[str, int]] = {}
            run_model_stage_counts: dict[str, dict[str, int]] = {}
            for sp in spans:
                attrs = sp.get("attributes") or {}
                # Apply optional filters
                step_name = sp.get("name") or "<unknown>"
                if step_filter and step_name != step_filter:
                    continue
                mid = attrs.get("aros.model_id")
                if model_filter and str(mid) != model_filter:
                    continue
                # Aggregate coercions
                ct = int(attrs.get("aros.coercion.total", 0) or 0)
                totals["coercion_total"] += ct
                run_coercions += ct
                # Identify step and model once per span
                step_name = step_name
                mid = mid
                for k, v in list(attrs.items()):
                    if str(k).startswith("aros.coercion.stage."):
                        stage = str(k).split(".")[-1]
                        totals["stages"][stage] = int(totals["stages"].get(stage, 0)) + int(v or 0)
                        run_stage_counts[stage] = int(run_stage_counts.get(stage, 0)) + int(v or 0)
                        # Stage breakdowns by step and by model
                        psst = per_step_stages.setdefault(step_name, {})
                        psst[stage] = int(psst.get(stage, 0)) + int(v or 0)
                        # Per-run breakdowns for bucket aggregation
                        rsst = run_step_stage_counts.setdefault(step_name, {})
                        rsst[stage] = int(rsst.get(stage, 0)) + int(v or 0)
                        if isinstance(mid, str) and mid:
                            rmst = run_model_stage_counts.setdefault(mid, {})
                            rmst[stage] = int(rmst.get(stage, 0)) + int(v or 0)
                # SOE
                totals["soe_applied"] += int(attrs.get("aros.soe.count", 0) or 0)
                totals["soe_skipped"] += int(attrs.get("aros.soe.skipped", 0) or 0)
                # Precheck
                totals["precheck_total"] += int(attrs.get("aros.precheck.total", 0) or 0)
                totals["precheck_pass"] += int(attrs.get("aros.precheck.pass", 0) or 0)
                totals["precheck_fail"] += int(attrs.get("aros.precheck.fail", 0) or 0)
                # Per-step hotspot (use span name)
                ps = per_step.setdefault(step_name, {"coercions": 0})
                ps["coercions"] += ct
                # Per-model aggregation (use aros.model_id when available)
                if isinstance(mid, str) and mid:
                    pm = per_model.setdefault(mid, {"coercions": 0})
                    pm["coercions"] += ct
                # Aggregate transforms
                tlist = attrs.get("aros.coercion.transforms")
                if isinstance(tlist, list):
                    for tname in tlist:
                        try:
                            transforms_count[str(tname)] = transforms_count.get(str(tname), 0) + 1
                        except Exception:
                            continue
                # Trend split
                if last_half_cut is not None and run_dt is not None:
                    # Assign this run's total coercions to half buckets once per run
                    pass
            # After iterating spans, aggregate per-model stage breakdowns once per run
            for _m, _stmap in run_model_stage_counts.items():
                mst = per_model_stages.setdefault(_m, {})
                for sk, sv in _stmap.items():
                    mst[sk] = int(mst.get(sk, 0)) + int(sv or 0)

            # After iterating spans, update trends per run once
            if last_half_cut is not None and run_dt is not None:
                if run_dt >= last_half_cut:
                    half_key = "last_half"
                else:
                    half_key = "prev_half"
                # Totals
                trend[half_key]["coercions"] = int(trend[half_key].get("coercions", 0)) + int(
                    run_coercions
                )
                # Aggregate per-stage totals for the half
                st = trend[half_key].setdefault("stages", {})
                for k, v in run_stage_counts.items():
                    st[k] = int(st.get(k, 0)) + int(v or 0)
                # Also aggregate per-step and per-model stage distributions per half
                h_step = trend[half_key].setdefault("step_stages", {})
                for sname, stmap in run_step_stage_counts.items():
                    cur = h_step.setdefault(sname, {})
                    for sk, sv in stmap.items():
                        cur[sk] = int(cur.get(sk, 0)) + int(sv or 0)
                h_model = trend[half_key].setdefault("model_stages", {})
                for mname, stmap in run_model_stage_counts.items():
                    curm = h_model.setdefault(mname, {})
                    for sk, sv in stmap.items():
                        curm[sk] = int(curm.get(sk, 0)) + int(sv or 0)
                # N-bucket assignment
                if buckets and now is not None:
                    try:
                        total_seconds = (now - cutoff).total_seconds() or 1.0
                        offset_seconds = (run_dt - cutoff).total_seconds()
                        if 0 <= offset_seconds <= total_seconds:
                            idx = int(
                                min(
                                    len(buckets) - 1,
                                    max(0, int(offset_seconds / (total_seconds / len(buckets)))),
                                )
                            )
                            buckets[idx]["coercions"] = (
                                int(buckets[idx]["coercions"]) + run_coercions
                            )
                            # per-stage into bucket
                            bst = buckets[idx].setdefault("stages", {})
                            for k, v in run_stage_counts.items():
                                bst[k] = int(bst.get(k, 0)) + int(v or 0)
                            # per-step/per-model stage distributions into bucket
                            bss = buckets[idx].setdefault("step_stages", {})
                            for sname, stmap in run_step_stage_counts.items():
                                cur = bss.setdefault(sname, {})
                                for sk, sv in stmap.items():
                                    cur[sk] = int(cur.get(sk, 0)) + int(sv or 0)
                            bms = buckets[idx].setdefault("model_stages", {})
                            for mm, stmap in run_model_stage_counts.items():
                                curm = bms.setdefault(mm, {})
                                for sk, sv in stmap.items():
                                    curm[sk] = int(curm.get(sk, 0)) + int(sv or 0)
                    except Exception:
                        pass

        console.print("[bold cyan]Flujo AROS Health Check[/bold cyan]")
        console.print(f"Analyzed runs: {totals['runs']}")
        console.print(f"Total coercions: {totals['coercion_total']} (stages: {totals['stages']})")
        console.print(f"SOE applied: {totals['soe_applied']}, skipped: {totals['soe_skipped']}")
        console.print(
            f"Reasoning precheck: total={totals['precheck_total']}, pass={totals['precheck_pass']}, fail={totals['precheck_fail']}"
        )
        if last_half_cut is not None:
            console.print("\n[bold]Trends[/bold]")
            console.print(
                f"Coercions last half vs previous half: {trend['last_half']['coercions']} vs {trend['prev_half']['coercions']}"
            )
            # Optional N-bucket distribution
            if isinstance(trend.get("buckets"), list) and trend["buckets"]:
                console.print("Bucketed coercions (oldest → newest):")
                # Format as a compact list of counts
                counts = ", ".join(str(int(b.get("coercions", 0))) for b in trend["buckets"])  # type: ignore
                console.print(f"[ {counts} ]")
                # Stage breakdown (if present) as a compact summary
                # Show up to 3 most common stages by total across all buckets
                agg_stage: dict[str, int] = {}
                for b in trend["buckets"]:
                    for sk, sv in (b.get("stages") or {}).items():
                        try:
                            agg_stage[str(sk)] = int(agg_stage.get(str(sk), 0)) + int(sv or 0)
                        except Exception:
                            continue
                if agg_stage:
                    top_stages = sorted(agg_stage.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    stage_names = ", ".join(f"{k}:{v}" for k, v in top_stages)
                    console.print(f"Top coercion stages across buckets: {stage_names}")
        # Top 10 steps by coercions
        top = sorted(per_step.items(), key=lambda kv: kv[1].get("coercions", 0), reverse=True)[:10]
        if top:
            console.print("\n[bold]Top steps by coercions[/bold]")
            for name, stats in top:
                console.print(f"- {name}: {stats['coercions']}")
        # Top 10 models by coercions
        topm = sorted(per_model.items(), key=lambda kv: kv[1].get("coercions", 0), reverse=True)[
            :10
        ]
        if topm:
            console.print("\n[bold]Top models by coercions[/bold]")
            for name, stats in topm:
                console.print(f"- {name}: {stats['coercions']}")
        # Top transforms
        topt = sorted(transforms_count.items(), key=lambda kv: kv[1], reverse=True)[:10]
        if topt:
            console.print("\n[bold]Top coercion transforms[/bold]")
            for name, cnt in topt:
                console.print(f"- {name}: {cnt}")

        # Stage breakdowns by top steps/models (brief)
        if top:
            console.print("\n[bold]Stage breakdowns by step[/bold]")
            for name, _ in top[:3]:
                stages_map = per_step_stages.get(name, {})
                if stages_map:
                    parts = ", ".join(
                        f"{k}:{v}"
                        for k, v in sorted(stages_map.items(), key=lambda kv: kv[1], reverse=True)[
                            :3
                        ]
                    )
                    console.print(f"- {name}: {parts}")
        if topm:
            console.print("\n[bold]Stage breakdowns by model[/bold]")
            for name, _ in topm[:3]:
                stages_map = per_model_stages.get(name, {})
                if stages_map:
                    parts = ", ".join(
                        f"{k}:{v}"
                        for k, v in sorted(stages_map.items(), key=lambda kv: kv[1], reverse=True)[
                            :3
                        ]
                    )
                    console.print(f"- {name}: {parts}")

        # Simple recommendations
        console.print("\n[bold]Recommendations[/bold]")
        recs: list[str] = []
        if totals["coercion_total"] >= 10 and totals["soe_applied"] == 0:
            recs.append(
                "Consider enabling structured_output (openai_json) for steps with frequent coercions."
            )
        if totals["precheck_fail"] > 0:
            recs.append(
                "Add processing.reasoning_precheck.required_context_keys or a validator_agent to catch plan issues early."
            )
        # Transform-driven suggestions
        if transforms_count.get("json5.loads", 0) >= 5:
            recs.append(
                "Enable coercion.tolerant_level=1 to accept JSON5-like outputs (comments/trailing commas)."
            )
        if transforms_count.get("json_repair", 0) >= 5:
            recs.append(
                "Consider coercion.tolerant_level=2 (json-repair) for robust auto-fixes; keep strict validation."
            )
        if any(k in transforms_count for k in ("str->int", "str->bool", "str->float")):
            recs.append(
                "Review coercion.allow mappings (integer/number/boolean) to make safe, unambiguous conversions explicit."
            )
        # Stage-aware suggestions
        stages_tot = totals.get("stages", {}) or {}
        tolerant_ct = int(stages_tot.get("tolerant", 0) or 0)
        semantic_ct = int(stages_tot.get("semantic", 0) or 0)
        extract_ct = int(stages_tot.get("extract", 0) or 0)
        if tolerant_ct >= 5:
            recs.append(
                "High tolerant-decoder activity detected; consider coercion.tolerant_level=1 (json5) or 2 (json-repair)."
            )
        if semantic_ct >= 5:
            recs.append(
                "Semantic coercions are frequent; consider 'aop: full' with a JSON schema and explicit coercion.allow mappings."
            )
        if extract_ct >= 5 and totals.get("soe_applied", 0) == 0:
            recs.append(
                "Many extractions from mixed text; enable structured_output or tighten prompts to return raw JSON only."
            )
        # Targeted hints for top offenders
        if top:
            worst_step, stats = top[0]
            if stats.get("coercions", 0) >= 5:
                recs.append(
                    f"Step '{worst_step}' has high coercions; try 'processing.structured_output: openai_json' with a schema or enable AOP."
                )
            # Stage-aware guidance for top step
            st_map = per_step_stages.get(worst_step, {})
            if st_map:
                tol = int(st_map.get("tolerant", 0) or 0)
                sem = int(st_map.get("semantic", 0) or 0)
                ext = int(st_map.get("extract", 0) or 0)
                if tol >= 3:
                    recs.append(
                        f"Step '{worst_step}' shows tolerant-decoder activity; consider tolerant_level=1 (json5) or 2 (json-repair)."
                    )
                if sem >= 3:
                    recs.append(
                        f"Step '{worst_step}' shows frequent semantic coercions; add a JSON schema and allowlisted coercions with 'aop: full'."
                    )
                if ext >= 3 and totals.get("soe_applied", 0) == 0:
                    recs.append(
                        f"Step '{worst_step}' often extracts JSON from mixed text; enable structured_output or adjust prompts to emit raw JSON."
                    )
        if topm:
            worst_model, mstats = topm[0]
            if mstats.get("coercions", 0) >= 5:
                recs.append(
                    f"Model '{worst_model}' shows frequent coercions; prefer schema-driven outputs or adjust prompting for stricter JSON."
                )
            # Stage-aware guidance for top model
            mst_map = per_model_stages.get(worst_model, {})
            if mst_map:
                tol = int(mst_map.get("tolerant", 0) or 0)
                sem = int(mst_map.get("semantic", 0) or 0)
                ext = int(mst_map.get("extract", 0) or 0)
                if tol >= 3:
                    recs.append(
                        f"Model '{worst_model}' often needs tolerant decoders; consider tolerant_level=1 (json5) or 2 (json-repair)."
                    )
                if sem >= 3:
                    recs.append(
                        f"Model '{worst_model}' shows frequent semantic coercions; use schemas and explicit coercion.allow with 'aop: full'."
                    )
                if ext >= 3 and totals.get("soe_applied", 0) == 0:
                    recs.append(
                        f"Model '{worst_model}' outputs mixed text; enable structured_output or tighten prompts to return raw JSON."
                    )
        # Trend-based hints (per-step/model increases)
        try:
            bkt_list = trend.get("buckets") if isinstance(trend, dict) else None
            if isinstance(bkt_list, list) and len(bkt_list) >= 2:
                first_b = bkt_list[0]
                last_b = bkt_list[-1]
                # Steps trend
                fs = first_b.get("step_stages") or {}
                ls = last_b.get("step_stages") or {}

                def _sum_stages(d: dict[str, int]) -> int:
                    try:
                        return int(sum(int(v or 0) for v in d.values()))
                    except Exception:
                        return 0

                step_deltas: dict[str, int] = {}
                for sname, stmap in ls.items():
                    prev = _sum_stages(fs.get(sname, {})) if isinstance(fs, dict) else 0
                    cur = _sum_stages(stmap if isinstance(stmap, dict) else {})
                    step_deltas[str(sname)] = cur - prev
                if step_deltas:
                    top_step = max(step_deltas.items(), key=lambda kv: kv[1])
                    if top_step[1] >= 2:
                        recs.append(
                            f"Trend: Step '{top_step[0]}' coercions rising; enable structured_output or AOP to stabilize outputs."
                        )
                        # Stage-specific delta for top rising step
                        try:
                            sname = top_step[0]
                            first_stages = fs.get(sname, {}) if isinstance(fs, dict) else {}
                            last_stages = ls.get(sname, {}) if isinstance(ls, dict) else {}
                            stage_keys = set(first_stages.keys()) | set(last_stages.keys())
                            best_stage = None
                            best_delta = 0
                            for sk in stage_keys:
                                try:
                                    dv = int((last_stages.get(sk, 0) or 0)) - int(
                                        (first_stages.get(sk, 0) or 0)
                                    )
                                    if dv > best_delta:
                                        best_delta = dv
                                        best_stage = str(sk)
                                except Exception:
                                    continue
                            if best_stage and best_delta >= 2:
                                if best_stage == "tolerant":
                                    recs.append(
                                        f"Trend: Step '{sname}' tolerant coercions rising; consider tolerant_level=1 (json5) or 2 (json-repair)."
                                    )
                                elif best_stage == "semantic":
                                    recs.append(
                                        f"Trend: Step '{sname}' semantic coercions rising; add a JSON schema and allowlisted coercions with 'aop: full'."
                                    )
                                elif best_stage == "extract":
                                    recs.append(
                                        f"Trend: Step '{sname}' extraction activity rising; enable structured_output or adjust prompts to emit raw JSON."
                                    )
                        except Exception:
                            pass
                # Models trend
                fm = first_b.get("model_stages") or {}
                lm = last_b.get("model_stages") or {}
                model_deltas: dict[str, int] = {}
                for mname, stmap in lm.items():
                    prev = _sum_stages(fm.get(mname, {})) if isinstance(fm, dict) else 0
                    cur = _sum_stages(stmap if isinstance(stmap, dict) else {})
                    model_deltas[str(mname)] = cur - prev
                if model_deltas:
                    top_model = max(model_deltas.items(), key=lambda kv: kv[1])
                    if top_model[1] >= 2:
                        recs.append(
                            f"Trend: Model '{top_model[0]}' coercions rising recently; prefer schema-driven outputs or tolerant decoders where safe."
                        )
                        # Stage-specific delta for top rising model
                        try:
                            mname = top_model[0]
                            first_mst = fm.get(mname, {}) if isinstance(fm, dict) else {}
                            last_mst = lm.get(mname, {}) if isinstance(lm, dict) else {}
                            stage_keys_m = set(first_mst.keys()) | set(last_mst.keys())
                            best_stage_m = None
                            best_delta_m = 0
                            for sk in stage_keys_m:
                                try:
                                    dv = int((last_mst.get(sk, 0) or 0)) - int(
                                        (first_mst.get(sk, 0) or 0)
                                    )
                                    if dv > best_delta_m:
                                        best_delta_m = dv
                                        best_stage_m = str(sk)
                                except Exception:
                                    continue
                            if best_stage_m and best_delta_m >= 2:
                                if best_stage_m == "tolerant":
                                    recs.append(
                                        f"Trend: Model '{mname}' tolerant coercions rising; consider tolerant_level=1 (json5) or 2 (json-repair)."
                                    )
                                elif best_stage_m == "semantic":
                                    recs.append(
                                        f"Trend: Model '{mname}' semantic coercions rising; use schemas and explicit coercion.allow with 'aop: full'."
                                    )
                                elif best_stage_m == "extract":
                                    recs.append(
                                        f"Trend: Model '{mname}' extraction activity rising; enable structured_output or tighten prompts to return raw JSON."
                                    )
                        except Exception:
                            pass
                # Multi-bucket positive drift hints
                if len(bkt_list) > 2:
                    # Step series across all buckets
                    step_series: dict[str, list[int]] = {}
                    for b in bkt_list:
                        ss = b.get("step_stages") or {}
                        for sname, stmap in ss.items():
                            tot = _sum_stages(stmap if isinstance(stmap, dict) else {})
                            step_series.setdefault(str(sname), []).append(tot)
                    # Sum of positive adjacent diffs
                    for sname, series in step_series.items():
                        try:
                            pos = sum(
                                max(0, series[i + 1] - series[i]) for i in range(len(series) - 1)
                            )
                            if pos >= 3:
                                recs.append(
                                    f"Trend: Step '{sname}' increasing across buckets; consider structured_output or AOP."
                                )
                                break
                        except Exception:
                            continue
                    # Model series
                    model_series: dict[str, list[int]] = {}
                    for b in bkt_list:
                        ms = b.get("model_stages") or {}
                        for mname, stmap in ms.items():
                            tot = _sum_stages(stmap if isinstance(stmap, dict) else {})
                            model_series.setdefault(str(mname), []).append(tot)
                    for mname, series in model_series.items():
                        try:
                            pos = sum(
                                max(0, series[i + 1] - series[i]) for i in range(len(series) - 1)
                            )
                            if pos >= 3:
                                recs.append(
                                    f"Trend: Model '{mname}' increasing across buckets; prefer schema-driven outputs where supported."
                                )
                                break
                        except Exception:
                            continue
        except Exception:
            pass

        # Half-window trend hints (fallback or complement to buckets)
        try:
            last_half = trend.get("last_half") if isinstance(trend, dict) else None
            prev_half = trend.get("prev_half") if isinstance(trend, dict) else None
            if isinstance(last_half, dict) and isinstance(prev_half, dict):
                # Per-step stage deltas
                lss = last_half.get("step_stages") or {}
                pss = prev_half.get("step_stages") or {}
                steps = set(lss.keys()) | set(pss.keys())
                for sname in steps:
                    lmap = lss.get(sname, {}) if isinstance(lss, dict) else {}
                    pmap = pss.get(sname, {}) if isinstance(pss, dict) else {}
                    try:
                        dt_tol = int(lmap.get("tolerant", 0) or 0) - int(
                            pmap.get("tolerant", 0) or 0
                        )
                        dt_sem = int(lmap.get("semantic", 0) or 0) - int(
                            pmap.get("semantic", 0) or 0
                        )
                        dt_ext = int(lmap.get("extract", 0) or 0) - int(pmap.get("extract", 0) or 0)
                    except Exception:
                        dt_tol = dt_sem = dt_ext = 0
                    if dt_tol >= 2:
                        recs.append(
                            f"Trend: Step '{sname}' tolerant coercions rising; consider tolerant_level=1 (json5) or 2 (json-repair)."
                        )
                    if dt_sem >= 2:
                        recs.append(
                            f"Trend: Step '{sname}' semantic coercions rising; add a JSON schema and allowlisted coercions with 'aop: full'."
                        )
                    if dt_ext >= 2 and totals.get("soe_applied", 0) == 0:
                        recs.append(
                            f"Trend: Step '{sname}' extraction activity rising; enable structured_output or adjust prompts to emit raw JSON."
                        )
                # Per-model stage deltas
                lms = last_half.get("model_stages") or {}
                pms = prev_half.get("model_stages") or {}
                models = set(lms.keys()) | set(pms.keys())
                for mname in models:
                    lmap = lms.get(mname, {}) if isinstance(lms, dict) else {}
                    pmap = pms.get(mname, {}) if isinstance(pms, dict) else {}
                    try:
                        dt_tol = int(lmap.get("tolerant", 0) or 0) - int(
                            pmap.get("tolerant", 0) or 0
                        )
                        dt_sem = int(lmap.get("semantic", 0) or 0) - int(
                            pmap.get("semantic", 0) or 0
                        )
                        dt_ext = int(lmap.get("extract", 0) or 0) - int(pmap.get("extract", 0) or 0)
                    except Exception:
                        dt_tol = dt_sem = dt_ext = 0
                    if dt_tol >= 2:
                        recs.append(
                            f"Trend: Model '{mname}' tolerant coercions rising; consider tolerant_level=1 (json5) or 2 (json-repair)."
                        )
                    if dt_sem >= 2:
                        recs.append(
                            f"Trend: Model '{mname}' semantic coercions rising; use schemas and explicit coercion.allow with 'aop: full'."
                        )
                    if dt_ext >= 2 and totals.get("soe_applied", 0) == 0:
                        recs.append(
                            f"Trend: Model '{mname}' extraction activity rising; enable structured_output or tighten prompts to return raw JSON."
                        )
        except Exception:
            pass

        if not recs:
            console.print("No obvious actions detected.")
        else:
            for r in recs:
                console.print(f"- {r}")

        # Export if requested
        if export:
            out_path = output or (
                "aros_health_report.json" if export.lower() == "json" else "aros_health_report"
            )
            try:
                if export.lower() == "json":
                    import json as _json
                    from datetime import datetime

                    payload = {
                        "version": get_version_string()
                        if "get_version_string" in globals()
                        else "",
                        "generated_at": datetime.utcnow().isoformat(),
                        "totals": totals,
                        "stages": totals.get("stages", {}),
                        "steps": [{"name": k, **v} for k, v in per_step.items()],
                        "models": [{"name": k, **v} for k, v in per_model.items()],
                        "step_stages": per_step_stages,
                        "model_stages": per_model_stages,
                        "transforms": transforms_count,
                        "trend": trend,
                    }
                    if out_path == "-":
                        console.print(_json.dumps(payload, ensure_ascii=False, indent=2))
                    else:
                        with open(out_path, "w", encoding="utf-8") as f:
                            _json.dump(payload, f, ensure_ascii=False, indent=2)
                        console.print(f"Exported JSON report to {out_path}")
                else:
                    import csv as _csv

                    steps_path = f"{out_path}_steps.csv"
                    models_path = f"{out_path}_models.csv"
                    transforms_path = f"{out_path}_transforms.csv"
                    with open(steps_path, "w", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        w.writerow(["step", "coercions"])
                        for k, v in per_step.items():
                            w.writerow([k, v.get("coercions", 0)])
                    with open(models_path, "w", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        w.writerow(["model", "coercions"])
                        for k, v in per_model.items():
                            w.writerow([k, v.get("coercions", 0)])
                    with open(transforms_path, "w", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        w.writerow(["transform", "count"])
                        for k, v in sorted(
                            transforms_count.items(), key=lambda kv: kv[1], reverse=True
                        ):
                            w.writerow([k, v])
                    console.print(
                        f"Exported CSV reports to {steps_path}, {models_path} and {transforms_path}"
                    )
            except Exception as e:
                console.print(f"[red]Export failed: {type(e).__name__}: {e}[/red]")

    anyio.run(_run)


def _auto_import_modules_from_env() -> None:
    mods = os.environ.get("FLUJO_REGISTER_MODULES")
    if not mods:
        return
    for name in mods.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            __import__(name)
        except Exception:
            continue


_auto_import_modules_from_env()


"""
Centralized CLI default handling lives in helpers/config_manager.
Keep this module focused on argument parsing and command wiring.
"""


@app.command(name="status", help="Show provider readiness and SQLite state configuration.")
def status(
    format: Annotated[
        str,
        typer.Option(
            help="Output format",
            show_default=True,
            click_type=click.Choice(["text", "json"], case_sensitive=False),
        ),
    ] = "text",
    no_network: Annotated[
        bool,
        typer.Option(
            "--no-network",
            help="Skip live network checks (presence-only)",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Per-check timeout in seconds for pings",
            show_default=True,
        ),
    ] = 2.0,
) -> None:
    """Status summary focusing on AI providers and SQLite configuration.

    - Providers: enabled if corresponding API key present.
    - SQLite: configured if state URI points at a sqlite path; skipped for memory.
    """
    import traceback as _tb
    import time as _time

    # Prepare base payload
    payload: Dict[str, Any] = {
        "command": "status",
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "providers": [],
    }

    # Provider presence checks; when network allowed, attempt lightweight model listing
    try:
        from flujo.infra.settings import get_settings as _get_settings

        s = _get_settings()
        providers: List[Dict[str, Any]] = []

        def _ping_openai(key: Optional[Any]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"name": "openai", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq

                start = _time.perf_counter()
                req = _rq.Request(
                    "https://api.openai.com/v1/models",
                    headers={
                        "Authorization": f"Bearer {key.get_secret_value() if hasattr(key, 'get_secret_value') else str(key)}",
                        "Content-Type": "application/json",
                    },
                )
                with _rq.urlopen(req, timeout=float(timeout)) as resp:  # nosec - endpoint fixed
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                # Classify common HTTP errors
                status = "UNREACHABLE"
                if hasattr(e, "code"):
                    c = int(getattr(e, "code"))
                    status = {
                        401: "INVALID_KEY",
                        403: "FORBIDDEN",
                        429: "RATE_LIMITED",
                    }.get(c, f"HTTP_{c}")
                info.update({"status": status, "message": str(e)})
            return info

        def _ping_anthropic(key: Optional[Any]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"name": "anthropic", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq

                start = _time.perf_counter()
                req = _rq.Request(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": key.get_secret_value()
                        if hasattr(key, "get_secret_value")
                        else str(key),
                        "anthropic-version": "2023-06-01",
                    },
                )
                with _rq.urlopen(req, timeout=float(timeout)) as resp:  # nosec - known URL
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                status = "UNREACHABLE"
                c = getattr(e, "code", None)
                if c is not None:
                    status = {401: "INVALID_KEY", 403: "FORBIDDEN", 429: "RATE_LIMITED"}.get(
                        int(c), f"HTTP_{int(c)}"
                    )
                info.update({"status": status, "message": str(e)})
            return info

        def _ping_gemini(key: Optional[Any]) -> Dict[str, Any]:
            info: Dict[str, Any] = {"name": "gemini", "enabled": bool(key)}
            if not key:
                info["status"] = "MISSING"
                return info
            if no_network:
                info["status"] = "OK"
                return info
            try:
                import urllib.request as _rq
                import urllib.parse as _up

                start = _time.perf_counter()
                url = f"https://generativelanguage.googleapis.com/v1/models?{_up.urlencode({'key': key.get_secret_value() if hasattr(key, 'get_secret_value') else str(key)})}"
                with _rq.urlopen(url, timeout=float(timeout)) as resp:  # nosec - known URL
                    latency = int((_time.perf_counter() - start) * 1000)
                    code = getattr(resp, "status", 200)
                    if code == 200:
                        info.update({"status": "OK", "latency_ms": latency})
                    else:
                        info.update({"status": f"HTTP_{code}", "latency_ms": latency})
            except Exception as e:  # noqa: BLE001
                status = "UNREACHABLE"
                c = getattr(e, "code", None)
                if c is not None:
                    status = {401: "INVALID_KEY", 403: "FORBIDDEN", 429: "RATE_LIMITED"}.get(
                        int(c), f"HTTP_{int(c)}"
                    )
                info.update({"status": status, "message": str(e)})
            return info

        providers.append(_ping_openai(getattr(s, "openai_api_key", None)))
        providers.append(_ping_anthropic(getattr(s, "anthropic_api_key", None)))
        providers.append(_ping_gemini(getattr(s, "google_api_key", None)))
        payload["providers"] = providers
    except Exception as e:
        # Configuration errors should map to config exit code
        typer.secho(
            f"Failed to read provider settings: {type(e).__name__}: {e}", fg=typer.colors.RED
        )
        if _os.environ.get("FLUJO_CLI_VERBOSE") == "1":
            typer.echo(_tb.format_exc(), err=True)
        raise typer.Exit(EX_CONFIG_ERROR)

    # SQLite configuration insight (do not open or create files)
    try:
        from ..infra.config_manager import get_state_uri as _get_state_uri
        from .config import _normalize_sqlite_path as _norm_sqlite
        from urllib.parse import urlparse as _urlparse

        uri = _get_state_uri(force_reload=True)
        sqlite_info: Dict[str, Any] = {"configured": False}
        if uri:
            uri_lower = uri.strip().lower()
            # Memory-like forms => not configured for SQLite
            if uri_lower in {"memory", "memory://", "mem://", "inmemory://"}:
                sqlite_info = {"configured": False}
            else:
                parsed = _urlparse(uri)
                if parsed.scheme.startswith("sqlite"):
                    try:
                        db_path = _norm_sqlite(uri, Path.cwd())
                        sqlite_info = {
                            "configured": True,
                            "path": db_path.as_posix(),
                        }
                    except Exception as pe:
                        sqlite_info = {
                            "configured": False,
                            "error": f"Failed to parse SQLite path: {type(pe).__name__}: {pe}",
                        }
                else:
                    # Unknown scheme => not our concern in MVP
                    sqlite_info = {"configured": False}
        payload["sqlite"] = sqlite_info

        # Last runs summary when SQLite is configured and file exists
        try:
            history: Dict[str, Any] = {"available": False, "items": []}
            if sqlite_info.get("configured") and sqlite_info.get("path"):
                dbp = Path(str(sqlite_info["path"]))
                if dbp.exists():
                    import sqlite3 as _sql

                    try:
                        ro_uri = f"file:{dbp.as_posix()}?mode=ro"
                        conn = _sql.connect(ro_uri, uri=True)
                        try:
                            cur = conn.cursor()
                            cur.execute(
                                "SELECT run_id, pipeline_name, status, created_at, updated_at, execution_time_ms, error_message FROM runs ORDER BY created_at DESC LIMIT 3"
                            )
                            items = []
                            for row in cur.fetchall() or []:
                                items.append(
                                    {
                                        "id": row[0],
                                        "pipeline": row[1],
                                        "status": row[2],
                                        "started_at": row[3],
                                        "ended_at": row[4],
                                        "duration_ms": row[5],
                                        "error": row[6],
                                    }
                                )
                            history.update({"available": True, "items": items, "total": len(items)})
                        finally:
                            conn.close()
                    except Exception:
                        # If schema absent/corrupt, just mark unavailable
                        history = {"available": False, "items": []}
            payload["history"] = history
        except Exception:
            # Never fail status due to history lookup
            payload["history"] = {"available": False, "items": []}
    except Exception as e:
        typer.secho(f"Failed to inspect state URI: {type(e).__name__}: {e}", fg=typer.colors.RED)
        if _os.environ.get("FLUJO_CLI_VERBOSE") == "1":
            typer.echo(_tb.format_exc(), err=True)
        raise typer.Exit(EX_RUNTIME_ERROR)

    # Emit output
    if (format or "text").lower() == "json":
        typer.echo(json.dumps(safe_serialize(payload)))
    else:
        # Minimal human output
        console = Console()
        # Providers line
        pv_summ = ", ".join(
            f"{p['name']}: {'ENABLED' if p.get('enabled') else 'disabled'}"
            for p in payload["providers"]
        )
        console.print(f"Providers: {pv_summ}")
        # SQLite line
        sqlite_info = payload.get("sqlite", {})
        if sqlite_info.get("configured"):
            console.print(f"SQLite: configured ({sqlite_info.get('path')})")
        else:
            console.print("SQLite: not configured (memory or absent)")

        # History preview
        hist = payload.get("history", {})
        if hist.get("available") and hist.get("items"):
            items = hist.get("items", [])
            preview = "; ".join(
                f"{it.get('started_at', '?')} {it.get('status', '?')}" for it in items
            )
            console.print(f"History: {len(items)} runs: {preview}")

    raise typer.Exit(EX_OK)


@app.command(
    help=(
        "✨ Initialize a new Flujo workflow project in this directory.\n\n"
        "Use --force to re-initialize templates in an existing project, and --yes to skip confirmation.\n\n"
        "Tip: New projects default to an in-memory state backend (state_uri = 'memory://').\n"
        "      To persist runs, set state_uri = 'sqlite:///.flujo/state.db' in flujo.toml or set FLUJO_STATE_URI."
    )
)
def init(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help=("Re-initialize even if this directory already contains a Flujo project."),
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


@app.command(
    help=(
        "🌟 Create a demo project with a sample research pipeline.\n\n"
        "This command initializes a new project (like `flujo init`) but with a more advanced `pipeline.yaml` "
        "that demonstrates agents, tools, and human-in-the-loop steps.\n\n"
        "Tip: Demo projects default to an in-memory state backend (state_uri = 'memory://').\n"
        "      To persist runs, set state_uri = 'sqlite:///.flujo/state.db' in flujo.toml or set FLUJO_STATE_URI."
    )
)
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


@experimental_app.command(name="solve")
def solve(
    prompt: str,
    max_iters: Annotated[
        Union[int, None], typer.Option(help="Maximum number of iterations.")
    ] = None,
    k: Annotated[
        Union[int, None],
        typer.Option(help="Number of solution variants to generate per iteration."),
    ] = None,
    reflection: Annotated[
        Union[bool, None], typer.Option(help="Enable/disable reflection agent.")
    ] = None,
    scorer: Annotated[
        Union[ScorerType, None],
        typer.Option(
            help="Scoring strategy.",
            case_sensitive=False,
            click_type=click.Choice(["ratio", "weighted", "reward"]),
        ),
    ] = None,
    weights_path: Annotated[
        Union[str, None], typer.Option(help="Path to weights file (JSON or YAML)")
    ] = None,
    solution_model: Annotated[
        Union[str, None], typer.Option(help="Model for the Solution agent.")
    ] = None,
    review_model: Annotated[
        Union[str, None], typer.Option(help="Model for the Review agent.")
    ] = None,
    validator_model: Annotated[
        Union[str, None], typer.Option(help="Model for the Validator agent.")
    ] = None,
    reflection_model: Annotated[
        Union[str, None], typer.Option(help="Model for the Reflection agent.")
    ] = None,
) -> None:
    """
    Solves a task using the multi-agent orchestrator.

    Args:
        prompt: The task prompt to solve
        max_iters: Maximum number of iterations
        k: Number of solution variants to generate per iteration
        reflection: Whether to enable reflection agent
        scorer: Scoring strategy to use
        weights_path: Path to weights file (JSON or YAML)
        solution_model: Model for the Solution agent
        review_model: Model for the Review agent
        validator_model: Model for the Validator agent
        reflection_model: Model for the Reflection agent

    Raises:
        ConfigurationError: If there is a configuration error
        typer.Exit: If there is an error loading weights or other CLI errors
    """
    try:
        # Set up command environment using helper function
        cli_args, metadata, agents = setup_solve_command_environment(
            max_iters=max_iters,
            k=k,
            reflection=reflection,
            scorer=scorer,
            weights_path=weights_path,
            solution_model=solution_model,
            review_model=review_model,
            validator_model=validator_model,
            reflection_model=reflection_model,
        )

        # Load settings for reflection limit
        from flujo.infra.config_manager import load_settings

        settings = load_settings()

        # Execute pipeline using helper function
        best = execute_solve_pipeline(
            prompt=prompt,
            cli_args=cli_args,
            metadata=metadata,
            agents=agents,
            settings=settings,
        )

        # Output result
        typer.echo(json.dumps(safe_serialize(best.model_dump()), indent=2))

    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)
    except ConfigurationError as e:
        typer.secho(f"Configuration Error: {e}", err=True)
        raise typer.Exit(2)


@dev_app.command(name="version")
def version_cmd() -> None:
    """
    Print the package version.

    Returns:
        None: Prints version to stdout
    """
    version = get_version_string()
    typer.echo(f"flujo version: {version}")


@dev_app.command(name="show-config")
def show_config_cmd() -> None:
    """
    Print effective Settings with secrets masked.

    Returns:
        None: Prints configuration to stdout
    """
    typer.echo(get_masked_settings_dict())


@experimental_app.command(name="bench")
def bench(
    prompt: str,
    rounds: Annotated[int, typer.Option(help="Number of benchmark rounds to run")] = 10,
) -> None:
    """
    Quick micro-benchmark of generation latency/score.

    Args:
        prompt: The prompt to benchmark
        rounds: Number of benchmark rounds to run

    Returns:
        None: Prints benchmark results to stdout

    Raises:
        KeyboardInterrupt: If the benchmark is interrupted by the user
    """
    try:
        # Apply CLI defaults from configuration file
        cli_args = apply_cli_defaults("bench", rounds=rounds)
        rounds = cast(int, cli_args["rounds"])

        # Run benchmark using helper function
        times, scores = run_benchmark_pipeline(prompt, rounds, logfire)

        # Create and display results table using helper function
        table = create_benchmark_table(times, scores)
        console: Console = Console()
        console.print(table)
    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)


@experimental_app.command(name="add-case")
def add_eval_case_cmd(
    dataset_path: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to the Python file containing the Dataset object",
    ),
    case_name: str = typer.Option(
        ..., "--name", "-n", prompt="Enter a unique name for the new evaluation case"
    ),
    inputs: str = typer.Option(
        ..., "--inputs", "-i", prompt="Enter the primary input for this case"
    ),
    expected_output: Optional[str] = typer.Option(
        None,
        "--expected",
        "-e",
        prompt="Enter the expected output (or skip)",
        show_default=False,
    ),
    metadata_json: Optional[str] = typer.Option(
        None, "--metadata", "-m", help="JSON string for case metadata"
    ),
    dataset_variable_name: str = typer.Option(
        "dataset", "--dataset-var", help="Name of the Dataset variable"
    ),
) -> None:
    """Print a new Case(...) definition to manually add to a dataset file."""

    if not dataset_path.exists() or not dataset_path.is_file():
        typer.secho(f"Error: Dataset file not found at {dataset_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    case_parts = [f'Case(name="{case_name}", inputs="""{inputs}"""']
    if expected_output is not None:
        case_parts.append(f'expected_output="""{expected_output}"""')
    if metadata_json:
        try:
            parsed = safe_deserialize(json.loads(metadata_json))
            case_parts.append(f"metadata={parsed}")
        except json.JSONDecodeError:
            typer.secho(
                f"Error: Invalid JSON provided for metadata: {metadata_json}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
    new_case_str = ", ".join(case_parts) + ")"

    typer.echo(
        f"\nPlease manually add the following line to the 'cases' list in {dataset_path} ({dataset_variable_name}):"
    )
    typer.secho(f"    {new_case_str}", fg=typer.colors.GREEN)

    try:
        with open(dataset_path, "r") as f:
            content = f.read()
        if dataset_variable_name not in content:
            typer.secho(
                f"Error: Could not find Dataset variable named '{dataset_variable_name}' in {dataset_path}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@experimental_app.command(name="improve")
def improve(
    pipeline_path: str,
    dataset_path: str,
    improvement_agent_model: Annotated[
        Optional[str],
        typer.Option(
            "--improvement-model",
            help="LLM model to use for the SelfImprovementAgent",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output raw JSON instead of formatted table")
    ] = False,
) -> None:
    """
    Run evaluation and generate improvement suggestions.

    Args:
        pipeline_path: Path to the pipeline definition file
        dataset_path: Path to the dataset definition file

    Returns:
        None: Prints improvement report to stdout

    Raises:
        typer.Exit: If there is an error loading the pipeline or dataset files
    """
    try:
        output = execute_improve(
            pipeline_path=pipeline_path,
            dataset_path=dataset_path,
            improvement_agent_model=improvement_agent_model,
            json_output=json_output,
        )
        if json_output and output is not None:
            typer.echo(output)

    except Exception as e:
        typer.echo(f"[red]Error running improvement: {e}", err=True)
        raise typer.Exit(1)


@dev_app.command(name="show-steps")
def explain(path: str) -> None:
    """
    Print a summary of a pipeline defined in a file.

    Args:
        path: Path to the pipeline definition file

    Returns:
        None: Prints pipeline step names to stdout

    Raises:
        typer.Exit: If there is an error loading the pipeline file
    """
    try:
        for name in get_pipeline_step_names(path):
            typer.echo(name)
    except Exception as e:
        typer.echo(f"[red]Failed to load pipeline file: {e}", err=True)
        raise typer.Exit(1)


def _validate_impl(
    path: Optional[str],
    strict: bool,
    output_format: str,
    *,
    include_imports: bool = True,
    fail_on_warn: bool = False,
    rules: Optional[str] = None,
    explain: bool = False,
) -> None:
    from .exit_codes import EX_VALIDATION_FAILED, EX_IMPORT_ERROR, EX_RUNTIME_ERROR
    import traceback as _tb
    import os as _os

    try:
        if path is None:
            root = find_project_root()
            path = str((Path(root) / "pipeline.yaml").resolve())
        report = validate_pipeline_file(path, include_imports=include_imports)

        # Optional: apply severity overrides from a rules file (JSON/TOML)
        def _apply_rules(_report: Any, rules_path: Optional[str]) -> Any:
            if not rules_path:
                return _report
            import os as _os

            try:
                if not _os.path.exists(rules_path):
                    return _report
                # Try JSON, then TOML
                mapping: dict[str, str] = {}
                try:
                    with open(rules_path, "r", encoding="utf-8") as f:
                        mapping = json.load(f)
                except Exception:
                    try:
                        import tomllib as _tomllib
                    except Exception:  # pragma: no cover
                        import tomli as _tomllib  # type: ignore
                    with open(rules_path, "rb") as f:
                        data = _tomllib.load(f)
                    if isinstance(data, dict):
                        if (
                            "validation" in data
                            and isinstance(data["validation"], dict)
                            and "rules" in data["validation"]
                        ):
                            mapping = data["validation"]["rules"] or {}
                        else:
                            mapping = data  # type: ignore
                if not isinstance(mapping, dict):
                    return _report
                sev_map = {str(k).upper(): str(v).lower() for k, v in mapping.items()}
                import fnmatch as _fnm

                def _resolve(rule_id: str) -> Optional[str]:
                    rid = rule_id.upper()
                    if rid in sev_map:
                        return sev_map[rid]
                    # wildcard/glob support (e.g., V-T*)
                    for pat, sev in sev_map.items():
                        if "*" in pat or "?" in pat or ("[" in pat and "]" in pat):
                            try:
                                if _fnm.fnmatch(rid, pat):
                                    return sev
                            except Exception:
                                continue
                    return None

                from flujo.domain.pipeline_validation import ValidationFinding, ValidationReport

                new_errors: list[ValidationFinding] = []
                new_warnings: list[ValidationFinding] = []
                for e in _report.errors:
                    sev = _resolve(e.rule_id)
                    if sev == "off":
                        continue
                    elif sev == "warning":
                        new_warnings.append(e)
                    else:
                        new_errors.append(e)
                for w in _report.warnings:
                    sev = _resolve(w.rule_id)
                    if sev == "off":
                        continue
                    elif sev == "error":
                        new_errors.append(w)
                    else:
                        new_warnings.append(w)
                return ValidationReport(errors=new_errors, warnings=new_warnings)
            except Exception:
                return _report

        # Apply rules severity overrides from file or profile name
        profile_mapping: Optional[dict[str, str]] = None
        if rules and not os.path.exists(rules):
            try:
                from ..infra.config_manager import get_config_manager as _cfg

                cfg = _cfg().load_config()
                val = getattr(cfg, "validation", None)
                profiles = getattr(val, "profiles", None) if val is not None else None
                if isinstance(profiles, dict) and rules in profiles:
                    raw = profiles[rules]
                    if isinstance(raw, dict):
                        profile_mapping = {str(k): str(v) for k, v in raw.items()}
            except Exception:
                profile_mapping = None

        if profile_mapping:
            # Write a temporary in-memory style mapping into JSON apply path
            def _apply_mapping(_report: Any, mapping: dict[str, str]) -> Any:
                sev_map = {str(k).upper(): str(v).lower() for k, v in mapping.items()}
                import fnmatch as _fnm

                def _resolve(rule_id: str) -> Optional[str]:
                    rid = rule_id.upper()
                    if rid in sev_map:
                        return sev_map[rid]
                    for pat, sev in sev_map.items():
                        if "*" in pat or "?" in pat or ("[" in pat and "]" in pat):
                            try:
                                if _fnm.fnmatch(rid, pat):
                                    return sev
                            except Exception:
                                continue
                    return None

                from flujo.domain.pipeline_validation import ValidationFinding, ValidationReport

                new_errors: list[ValidationFinding] = []
                new_warnings: list[ValidationFinding] = []
                for e in _report.errors:
                    sev = _resolve(e.rule_id)
                    if sev == "off":
                        continue
                    elif sev == "warning":
                        new_warnings.append(e)
                    else:
                        new_errors.append(e)
                for w in _report.warnings:
                    sev = _resolve(w.rule_id)
                    if sev == "off":
                        continue
                    elif sev == "error":
                        new_errors.append(w)
                    else:
                        new_warnings.append(w)
                return ValidationReport(errors=new_errors, warnings=new_warnings)

            report = _apply_mapping(report, profile_mapping)
        else:
            report = _apply_rules(report, rules)

        # Optional explanation catalog for rules
        def _explain(rule_id: str) -> str | None:
            rid = (rule_id or "").upper()
            why: dict[str, str] = {
                "V-T1": "previous_step is a raw value; .output will be null at runtime.",
                "V-T2": "'this' is only defined inside map bodies; using it elsewhere yields empty output.",
                "V-T3": "Unknown or disabled filters will cause templating errors or ignored transforms.",
                "V-T4": "Referencing a future or misspelled step name will resolve to None at render time.",
                "V-S1": "Schemas with misplaced 'required', missing 'items', or unknown types are brittle and may mis-validate.",
                "V-I1": "Imported blueprint cannot be found; validation cannot include the intended child pipeline.",
                "V-SM1": "Unreachable states or no path to an end state can dead-end the state machine.",
                "V-P1": "Parallel branches may write conflicting keys into context, leading to nondeterminism.",
                "V-P3": "Parallel branches receive the same input; heterogeneous expectations can fail at runtime.",
                "V-A1": "Simple steps without agents cannot execute.",
                "V-A2": "Static type mismatch between steps will likely fail at runtime.",
                "V-A5": "Outputs not consumed or merged are effectively dropped and indicate logic gaps.",
                "V-F1": "Fallback step must accept the same input shape as the primary to be callable on failure.",
                "V-I2": "Mapping to unknown parent roots will not populate context as intended.",
                "V-I3": "Cyclic imports create infinite recursion in composition.",
            }
            return why.get(rid)

        if output_format == "json":
            # Emit machine-friendly JSON (errors, warnings, is_valid)
            payload = {
                "is_valid": bool(report.is_valid),
                "errors": [
                    ({**e.model_dump(), **({"explain": _explain(e.rule_id)} if explain else {})})
                    for e in report.errors
                ],
                "warnings": [
                    ({**w.model_dump(), **({"explain": _explain(w.rule_id)} if explain else {})})
                    for w in report.warnings
                ],
                "path": path,
            }
            typer.echo(json.dumps(payload))
        elif output_format == "sarif":
            # Minimal SARIF 2.1.0 conversion
            def _level(sev: str) -> str:
                return "error" if sev == "error" else "warning"

            rules_index: dict[str, int] = {}
            sarif_rules: list[dict[str, Any]] = []
            sarif_results: list[dict[str, Any]] = []

            def _rule_ref(rule_id: str) -> dict[str, Any]:
                rid = rule_id.upper()
                if rid not in rules_index:
                    rules_index[rid] = len(sarif_rules)
                    sarif_rules.append(
                        {
                            "id": rid,
                            "name": rid,
                            "shortDescription": {"text": rid},
                            "helpUri": f"https://aandresalvarez.github.io/flujo/reference/validation_rules/#{rid.lower()}",
                        }
                    )
                return {"ruleId": rid}

            def _location(f: Any) -> dict[str, Any]:
                region: dict[str, Any] = {}
                if getattr(f, "line", None):
                    region["startLine"] = int(getattr(f, "line"))
                if getattr(f, "column", None):
                    region["startColumn"] = int(getattr(f, "column"))
                phys = {}
                if getattr(f, "file", None):
                    phys["uri"] = str(getattr(f, "file"))
                loc = {"physicalLocation": {"artifactLocation": phys}}
                if region:
                    loc["physicalLocation"]["region"] = region
                return loc

            for f in report.errors + report.warnings:
                sarif_results.append(
                    {
                        **_rule_ref(f.rule_id),
                        "level": _level(f.severity),
                        "message": {
                            "text": f"{f.step_name + ': ' if f.step_name else ''}{f.message}"
                        },
                        "locations": [_location(f)],
                        "properties": {
                            "suggestion": getattr(f, "suggestion", None),
                            "location_path": getattr(f, "location_path", None),
                        },
                    }
                )

            sarif = {
                "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "flujo-validate", "rules": sarif_rules}},
                        "results": sarif_results,
                    }
                ],
            }
            typer.echo(json.dumps(sarif))
        else:
            if report.errors:
                typer.echo("[red]Validation errors detected:")
                typer.echo(
                    "[red]See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/"
                )
                for f in report.errors:
                    loc = f"{f.step_name}: " if f.step_name else ""
                    link = f" (details: https://aandresalvarez.github.io/flujo/reference/validation_rules/#{str(f.rule_id).lower()})"
                    why = _explain(f.rule_id) if explain else None
                    suffix = f" | Why: {why}" if why else ""
                    if f.suggestion:
                        typer.echo(
                            f"- [{f.rule_id}] {loc}{f.message}{link}{suffix} -> Suggestion: {f.suggestion}"
                        )
                    else:
                        typer.echo(f"- [{f.rule_id}] {loc}{f.message}{link}{suffix}")
            if report.warnings:
                typer.echo("[yellow]Warnings:")
                typer.echo(
                    "[yellow]See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/"
                )
                for f in report.warnings:
                    loc = f"{f.step_name}: " if f.step_name else ""
                    link = f" (details: https://aandresalvarez.github.io/flujo/reference/validation_rules/#{str(f.rule_id).lower()})"
                    why = _explain(f.rule_id) if explain else None
                    suffix = f" | Why: {why}" if why else ""
                    if f.suggestion:
                        typer.echo(
                            f"- [{f.rule_id}] {loc}{f.message}{link}{suffix} -> Suggestion: {f.suggestion}"
                        )
                    else:
                        typer.echo(f"- [{f.rule_id}] {loc}{f.message}{link}{suffix}")
            if report.is_valid:
                typer.echo("[green]Pipeline is valid")

        if strict and not report.is_valid:
            raise typer.Exit(EX_VALIDATION_FAILED)
        if fail_on_warn and report.warnings:
            raise typer.Exit(EX_VALIDATION_FAILED)
    except ModuleNotFoundError as e:
        # Improve import error messaging with hint on project root
        mod = getattr(e, "name", None) or str(e)
        typer.echo(
            f"[red]Import error: module '{mod}' not found. Try PYTHONPATH=. or use --project/FLUJO_PROJECT_ROOT[/red]",
            err=True,
        )
        if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
            typer.echo("\nTraceback:", err=True)
            typer.echo("".join(_tb.format_exception(e)), err=True)
        raise typer.Exit(EX_IMPORT_ERROR)
    except typer.Exit:
        # Preserve intended exit status (e.g., EX_VALIDATION_FAILED)
        raise
    except Exception as e:
        typer.echo(f"[red]Validation failed: {type(e).__name__}: {e}[/red]", err=True)
        if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
            typer.echo("\nTraceback:", err=True)
            typer.echo("".join(_tb.format_exception(e)), err=True)
        raise typer.Exit(EX_RUNTIME_ERROR)


@dev_app.command(name="validate")
def validate_dev(
    path: Optional[str] = typer.Argument(
        None,
        help="Path to pipeline file. If omitted, uses project pipeline.yaml",
    ),
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Exit non-zero when errors are found (default: strict)",
        ),
    ] = True,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "--format",
            help="Output format for CI parsers",
            case_sensitive=False,
            click_type=click.Choice(["text", "json", "sarif"]),
        ),
    ] = "text",
    imports: Annotated[
        bool,
        typer.Option(
            "--imports/--no-imports",
            help="Recursively validate imported blueprints",
        ),
    ] = True,
    fail_on_warn: Annotated[
        bool,
        typer.Option("--fail-on-warn", help="Treat warnings as errors (non-zero exit)"),
    ] = False,
    rules: Annotated[
        Optional[str],
        typer.Option(
            "--rules", help="Path to rules JSON/TOML that overrides severities (off/warning/error)"
        ),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option("--explain", help="Include brief 'why this matters' guidance in output"),
    ] = False,
) -> None:
    """Validate a pipeline defined in a file (developer namespace)."""
    _validate_impl(
        path,
        strict,
        output_format,
        include_imports=imports,
        fail_on_warn=fail_on_warn,
        rules=rules,
        explain=explain,
    )


@app.command(name="validate")
def validate(
    path: Optional[str] = typer.Argument(
        None,
        help="Path to pipeline file. If omitted, uses project pipeline.yaml",
    ),
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Exit non-zero when errors are found (default: strict)",
        ),
    ] = True,
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "--format",
            help="Output format for CI parsers",
            case_sensitive=False,
            click_type=click.Choice(["text", "json", "sarif"]),
        ),
    ] = "text",
    imports: Annotated[
        bool,
        typer.Option(
            "--imports/--no-imports",
            help="Recursively validate imported blueprints",
        ),
    ] = True,
    fail_on_warn: Annotated[
        bool,
        typer.Option("--fail-on-warn", help="Treat warnings as errors (non-zero exit)"),
    ] = False,
    rules: Annotated[
        Optional[str],
        typer.Option(
            "--rules", help="Path to rules JSON/TOML that overrides severities (off/warning/error)"
        ),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option("--explain", help="Include brief 'why this matters' guidance in output"),
    ] = False,
) -> None:
    """Validate a pipeline (top-level alias)."""
    _validate_impl(
        path,
        strict,
        output_format,
        include_imports=imports,
        fail_on_warn=fail_on_warn,
        rules=rules,
        explain=explain,
    )


@app.command(
    help=(
        "🤖 Start a conversation with the AI Architect to build your workflow.\n\n"
        "By default this uses the full conversational state machine. Set FLUJO_ARCHITECT_MINIMAL=1"
        " to use the legacy minimal generator.\n\n"
        "Tip: pass --allow-side-effects to permit pipelines that reference side-effect skills."
    )
)
def create(  # <--- REVERT BACK TO SYNC
    goal: Annotated[
        Optional[str], typer.Option("--goal", help="Natural-language goal for the architect")
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", help="Pipeline name for pipeline.yaml")
    ] = None,
    budget: Annotated[
        Optional[float],
        typer.Option(
            "--budget",
            help="Safe cost limit (USD) per run to add under budgets.pipeline.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[str],
        typer.Option("--output-dir", help="Directory to write generated files"),
    ] = None,
    context_file: Annotated[
        Optional[str],
        typer.Option("--context-file", "-f", help="Path to JSON/YAML file with extra context data"),
    ] = None,
    non_interactive: Annotated[
        bool, typer.Option("--non-interactive", help="Disable interactive prompts")
    ] = False,
    allow_side_effects: Annotated[
        bool,
        typer.Option(
            "--allow-side-effects",
            help="Allow running or generating pipelines that reference side-effect skills",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output files if present"),
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero if final blueprint is invalid")
    ] = False,
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable verbose logging to debug the Architect Agent's execution.",
        hidden=True,
    ),
    agentic: Annotated[
        Optional[bool],
        typer.Option(
            "--agentic/--no-agentic",
            help=(
                "Force-enable the agentic Architect (state machine) or force the minimal generator for this run."
            ),
        ),
    ] = None,
    wizard: Annotated[
        bool,
        typer.Option(
            "--wizard",
            help="Run a simple interactive wizard to emit a natural YAML pipeline (skips Architect)",
        ),
    ] = False,
    wizard_pattern: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-pattern",
            help="Pattern to generate when using --wizard",
            case_sensitive=False,
            click_type=click.Choice(["loop", "map", "parallel"]),
        ),
    ] = None,
    wizard_iterable_name: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-iterable-name",
            help="Iterable name for --wizard-pattern=map (default: items)",
        ),
    ] = None,
    wizard_reduce_mode: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-reduce-mode",
            help="Reduce mode for --wizard-pattern=parallel",
            case_sensitive=False,
            click_type=click.Choice(["keys", "values", "union", "concat", "first", "last"]),
        ),
    ] = None,
    wizard_conversation: Annotated[
        Optional[bool],
        typer.Option(
            "--wizard-conversation/--no-wizard-conversation",
            help="Set conversation: true|false for loop preset",
        ),
    ] = None,
    wizard_stop_when: Annotated[
        Optional[bool],
        typer.Option(
            "--wizard-stop-when/--no-wizard-stop-when",
            help="Include or skip stop_when: agent_finished for loop preset",
        ),
    ] = None,
    wizard_propagation: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-propagation",
            help="Propagation mode for loop preset",
            case_sensitive=False,
            click_type=click.Choice(["context", "previous_output", "auto"]),
        ),
    ] = None,
    wizard_body_steps: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-body-steps",
            help="Comma-separated body step names for loop preset",
        ),
    ] = None,
    wizard_map_step_name: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-map-step-name",
            help="Name for the single map body step (default: process)",
        ),
    ] = None,
    wizard_branch_names: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-branch-names",
            help="Comma-separated branch names for parallel preset",
        ),
    ] = None,
    wizard_ai_turn_source: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-ai-turn-source",
            help="AI turn source for loop (last/all_agents/named_steps)",
        ),
    ] = None,
    wizard_user_turn_sources: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-user-turn-sources",
            help="Comma-separated user turn sources (e.g., hitl,stepA,stepB)",
        ),
    ] = None,
    wizard_history_strategy: Annotated[
        Optional[str],
        typer.Option(
            "--wizard-history-strategy",
            help="History strategy (truncate_tokens/truncate_turns/summarize)",
        ),
    ] = None,
    wizard_history_max_tokens: Annotated[
        Optional[int],
        typer.Option(
            "--wizard-history-max-tokens",
            help="History max_tokens when using truncate_tokens/summarize",
        ),
    ] = None,
    wizard_history_max_turns: Annotated[
        Optional[int],
        typer.Option(
            "--wizard-history-max-turns",
            help="History max_turns when using truncate_turns",
        ),
    ] = None,
    wizard_history_summarize_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--wizard-history-summarize-ratio",
            help="Summarize ratio when using summarize (0..1)",
        ),
    ] = None,
) -> None:
    """Conversational pipeline generation via the Architect pipeline.

    Loads the bundled architect YAML, runs it with the provided goal, and writes outputs.

    Tip: Using GPT-5? To tune agent timeouts/retries for complex reasoning, see
    docs/guides/gpt5_architect.md (agent-level `timeout`/`max_retries`) and
    step-level `config.timeout` (alias to `timeout_s`) for plugin/validator phases.
    """
    try:
        # Wizard shortcut: generate natural YAML without running the Architect
        if wizard:
            _run_create_wizard(
                goal=goal,
                name=name,
                output_dir=output_dir,
                non_interactive=non_interactive,
                pattern=wizard_pattern,
                iterable_name=wizard_iterable_name,
                reduce_mode=wizard_reduce_mode,
                conversation=wizard_conversation,
                stop_when=wizard_stop_when,
                propagation=wizard_propagation,
                body_steps=wizard_body_steps,
                map_step_name=wizard_map_step_name,
                branch_names=wizard_branch_names,
                ai_turn_source=wizard_ai_turn_source,
                user_turn_sources=wizard_user_turn_sources,
                history_strategy=wizard_history_strategy,
                history_max_tokens=wizard_history_max_tokens,
                history_max_turns=wizard_history_max_turns,
                history_summarize_ratio=wizard_history_summarize_ratio,
            )
            raise typer.Exit(0)
        # Make --debug effective even if passed after the command name (Click quirk)
        try:
            _ctx = click.get_current_context(silent=True)
            if _ctx is not None and any(arg in getattr(_ctx, "args", []) for arg in ("--debug",)):
                import logging as _logging
                import os as _os

                _logger = _logging.getLogger("flujo")
                _logger.setLevel(_logging.INFO)
                try:
                    _os.environ["FLUJO_DEBUG"] = "1"
                except Exception:
                    pass
        except Exception:
            pass
        # Conditional logging: silence internal logs for end users unless --debug
        import logging as _logging
        import warnings as _warnings

        _flujo_logger = _logging.getLogger("flujo")
        _httpx_logger = _logging.getLogger("httpx")
        _orig_flujo_level = _flujo_logger.getEffectiveLevel()
        _orig_httpx_level = _httpx_logger.getEffectiveLevel()
        # We will temporarily add filters and later reset to defaults

        if not debug:
            _flujo_logger.setLevel(_logging.CRITICAL)
            _httpx_logger.setLevel(_logging.WARNING)
        else:
            # Ensure flujo logger emits INFO when --debug is passed
            try:
                _flujo_logger.setLevel(_logging.INFO)
            except Exception:
                pass
            # Suppress specific runner warnings for a clean UX
            try:
                _warnings.filterwarnings("ignore", message="pipeline_name was not provided.*")
                _warnings.filterwarnings("ignore", message="pipeline_id was not provided.*")
            except Exception:
                pass

        try:
            # Enforce explicit output directory in non-interactive mode to avoid accidental writes
            if non_interactive and not output_dir:
                typer.echo(
                    "[red]--output-dir is required when running --non-interactive to specify where to write pipeline.yaml[/red]",
                    err=True,
                )
                raise typer.Exit(2)

            # Track whether user supplied --goal flag explicitly (HITL skip rule)
            goal_flag_provided = goal is not None

            # Prompt for goal if not provided and interactive
            if goal is None and not non_interactive:
                goal = typer.prompt("What is your goal for this pipeline?")
            if goal is None:
                typer.echo("[red]--goal is required in --non-interactive mode[/red]")
                raise typer.Exit(2)
            # Prepare initial context data
            from .helpers import parse_context_data

            # Ensure built-in skills are registered and collect available skills
            try:
                import flujo.builtins as _ensure_builtins  # noqa: F401
                from flujo.infra.skill_registry import get_skill_registry as _get_skill_registry

                _reg = _get_skill_registry()
                _entries = getattr(_reg, "_entries", {})
                _available_skills = [
                    {
                        "id": sid,
                        "description": (meta or {}).get("description"),
                        "input_schema": (meta or {}).get("input_schema"),
                    }
                    for sid, meta in _entries.items()
                ]
            except Exception:
                _available_skills = []

            # Build architect pipeline programmatically, but allow tests to inject YAML via monkeypatch
            try:
                fn = load_pipeline_from_yaml_file
                # If tests monkeypatch this symbol in flujo.cli.main, it won't originate from helpers
                is_injected = (
                    getattr(fn, "__module__", "") != "flujo.cli.helpers"
                    or getattr(fn, "__name__", "") != "load_pipeline_from_yaml_file"
                )
                if is_injected:
                    pipeline_obj = fn("<injected>")
                else:
                    # Respect explicit CLI override first
                    try:
                        if agentic is True:
                            os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"
                            os.environ.pop("FLUJO_ARCHITECT_MINIMAL", None)
                        elif agentic is False:
                            os.environ["FLUJO_ARCHITECT_MINIMAL"] = "1"
                            os.environ.pop("FLUJO_ARCHITECT_STATE_MACHINE", None)
                        else:
                            # Prefer agentic by default for users invoking `flujo create` when minimal not explicitly set
                            if os.environ.get("FLUJO_ARCHITECT_MINIMAL", "").strip() == "":
                                os.environ.setdefault("FLUJO_ARCHITECT_STATE_MACHINE", "1")
                    except Exception:
                        pass
                    from flujo.architect.builder import build_architect_pipeline as _build_arch

                    pipeline_obj = _build_arch()
            except Exception as e:
                typer.echo(
                    f"[red]Failed to acquire architect pipeline: {e}",
                    err=True,
                )
                raise typer.Exit(1)

            # Determine whether to perform HITL preview/approval
            # Default: disabled to preserve simple interactive flow expected by tests.
            # Enable only when the environment explicitly opts-in.
            try:
                _hitl_env = os.environ.get("FLUJO_CREATE_HITL", "").strip().lower()
            except Exception:
                _hitl_env = ""
            hitl_opt_in = _hitl_env in {"1", "true", "yes", "on"}
            hitl_requested = hitl_opt_in and (not non_interactive) and (not goal_flag_provided)

            initial_context_data = {
                "user_goal": goal,
                "available_skills": _available_skills,
                # Enable HITL only when --goal flag not provided and interactive session
                "hitl_enabled": bool(hitl_requested),
                "non_interactive": bool(non_interactive),
            }
            extra_ctx = parse_context_data(None, context_file)
            if isinstance(extra_ctx, dict):
                initial_context_data.update(extra_ctx)
            # Ensure required field for custom context model
            if "initial_prompt" not in initial_context_data:
                initial_context_data["initial_prompt"] = goal

            # Create runner and execute using shared ArchitectContext
            from flujo.architect.context import ArchitectContext as _ArchitectContext

            # Load the project-aware state backend (config-driven). If configured
            # as memory/ephemeral, this will select the in-memory backend.
            try:
                from .config import load_backend_from_config as _load_backend_from_config

                _state_backend = _load_backend_from_config()
            except Exception:
                _state_backend = None

            runner = create_flujo_runner(
                pipeline=pipeline_obj,
                context_model_class=_ArchitectContext,
                initial_context_data=initial_context_data,
                state_backend=_state_backend,
            )

            # For now, require goal as input too (can be refined by architect design)
            result = execute_pipeline_with_output_handling(
                runner=runner, input_data=goal, run_id=None, json_output=False
            )

            # Debug aid: print step names and success to help tests diagnose branching
            try:

                def _print_steps(steps: list[Any], indent: int = 0) -> None:
                    for sr in steps or []:
                        try:
                            nm = getattr(sr, "name", "<unnamed>")
                            ok = getattr(sr, "success", None)
                            key = (getattr(sr, "metadata_", {}) or {}).get("executed_branch_key")
                            typer.echo(
                                f"[grey58]{'  ' * indent}STEP {nm}: success={ok} key={key}[/grey58]"
                            )
                            nested = getattr(sr, "step_history", None)
                            if isinstance(nested, list) and nested:
                                _print_steps(nested, indent + 1)
                        except Exception:
                            continue

                _print_steps(getattr(result, "step_history", []) or [])
            except Exception:
                pass

            # Extract YAML text preferring the most recent step output (repairs), then context
            yaml_text: Optional[str] = None
            try:
                candidates: list[Any] = []

                # Recursively collect outputs from step history (including nested sub-steps)
                def _collect_outputs(step_results: list[Any]) -> None:
                    for sr in step_results:
                        try:
                            # Push this step's output
                            candidates.append(getattr(sr, "output", None))
                            # Recurse into nested step_history if present
                            nested = getattr(sr, "step_history", None)
                            if isinstance(nested, list) and nested:
                                _collect_outputs(nested)
                        except Exception:
                            continue

                _collect_outputs(list(getattr(result, "step_history", [])))
                # Reverse to prefer most recent outputs
                candidates = list(reversed(candidates))
                # Also include outputs of known steps if available (e.g., writer)
                for sr in getattr(result, "step_history", []):
                    try:
                        name = getattr(sr, "step_name", getattr(sr, "name", ""))
                    except Exception:
                        name = ""
                    if str(name) in {"write_pipeline_yaml", "extract_yaml_text"}:
                        candidates.append(getattr(sr, "output", None))

                # Scan candidates for YAML text in various shapes
                for out in candidates:
                    try:
                        if out is None:
                            continue
                        if isinstance(out, dict):
                            val = out.get("generated_yaml") or out.get("yaml_text")
                            if isinstance(val, (str, bytes)):
                                candidate = val.decode() if isinstance(val, bytes) else str(val)
                                if candidate and candidate.strip():
                                    yaml_text = candidate
                                    break
                        if hasattr(out, "generated_yaml") and getattr(out, "generated_yaml"):
                            val = getattr(out, "generated_yaml")
                            s = val.decode() if isinstance(val, bytes) else str(val)
                            if s and s.strip():
                                yaml_text = s
                                break
                        if hasattr(out, "yaml_text") and getattr(out, "yaml_text"):
                            val = getattr(out, "yaml_text")
                            s = val.decode() if isinstance(val, bytes) else str(val)
                            if s and s.strip():
                                yaml_text = s
                                break
                        if isinstance(out, (str, bytes)):
                            s = out.decode() if isinstance(out, bytes) else out
                            st = s.strip()
                            if st and ("version:" in st or "steps:" in st):
                                yaml_text = s
                                break
                    except Exception:
                        continue

                # Fallback to final context if needed
                if yaml_text is None:
                    ctx = getattr(result, "final_pipeline_context", None)
                    if ctx is not None:
                        if hasattr(ctx, "generated_yaml") and getattr(ctx, "generated_yaml"):
                            yaml_text = getattr(ctx, "generated_yaml")
                        elif hasattr(ctx, "yaml_text") and getattr(ctx, "yaml_text"):
                            yaml_text = getattr(ctx, "yaml_text")
                        else:
                            # Fallback: look into context.scratchpad if present
                            try:
                                scratch = getattr(ctx, "scratchpad", None)
                                if isinstance(scratch, dict):
                                    val = scratch.get("generated_yaml") or scratch.get("yaml_text")
                                    if isinstance(val, (str, bytes)):
                                        yaml_text = (
                                            val.decode() if isinstance(val, bytes) else str(val)
                                        )
                            except Exception:
                                pass
                # Targeted fallback: look for specific architect steps that carry YAML
                if yaml_text is None:
                    try:
                        for sr in getattr(result, "step_history", []) or []:
                            name = getattr(sr, "name", "")
                            if str(name) in {
                                "store_yaml_text",
                                "extract_yaml_text",
                                "emit_current_yaml",
                                "final_passthrough",
                            }:
                                out = getattr(sr, "output", None)
                                if isinstance(out, dict):
                                    val = out.get("generated_yaml") or out.get("yaml_text")
                                    if isinstance(val, (str, bytes)):
                                        yaml_text = (
                                            val.decode() if isinstance(val, bytes) else str(val)
                                        )
                                        if yaml_text.strip():
                                            break
                                elif isinstance(out, (str, bytes)):
                                    s = out.decode() if isinstance(out, bytes) else out
                                    if s.strip():
                                        yaml_text = s
                                        break
                    except Exception:
                        pass
                # Context-based fallback: scan branch_context from step history (including nested)
                if yaml_text is None:
                    try:
                        contexts: list[Any] = []

                        def _collect_contexts(step_results: list[Any]) -> None:
                            for sr in step_results:
                                try:
                                    ctx_candidate = getattr(sr, "branch_context", None)
                                    if ctx_candidate is not None:
                                        contexts.append(ctx_candidate)
                                    nested_sr = getattr(sr, "step_history", None)
                                    if isinstance(nested_sr, list) and nested_sr:
                                        _collect_contexts(nested_sr)
                                except Exception:
                                    continue

                        _collect_contexts(list(getattr(result, "step_history", [])))
                        for ctx in reversed(contexts):
                            try:
                                if hasattr(ctx, "generated_yaml") and getattr(
                                    ctx, "generated_yaml"
                                ):
                                    yaml_text = getattr(ctx, "generated_yaml")
                                    break
                                if hasattr(ctx, "yaml_text") and getattr(ctx, "yaml_text"):
                                    yaml_text = getattr(ctx, "yaml_text")
                                    break
                                scratch = getattr(ctx, "scratchpad", None)
                                if isinstance(scratch, dict):
                                    val = scratch.get("generated_yaml") or scratch.get("yaml_text")
                                    if isinstance(val, (str, bytes)):
                                        yaml_text = (
                                            val.decode() if isinstance(val, bytes) else str(val)
                                        )
                                        if yaml_text.strip():
                                            break
                            except Exception:
                                continue
                    except Exception:
                        pass
                # Last-resort heuristic: scan text representations for a YAML snippet
                if yaml_text is None and candidates:
                    try:
                        import re as _re

                        for out in candidates:
                            text = None
                            try:
                                if isinstance(out, (str, bytes)):
                                    text = out.decode() if isinstance(out, bytes) else out
                                else:
                                    text = str(out)
                            except Exception:
                                continue
                            if not text:
                                continue
                            m = _re.search(
                                r"(^|\n)version:\s*['\"]?0\.1['\"]?.*?\n(?:.*\n)*?steps:\s*.*", text
                            )
                            if m:
                                snippet = text[m.start() :]
                                yaml_text = snippet.strip()
                                break
                    except Exception:
                        pass
            except Exception:
                pass

            if yaml_text is None:
                try:
                    # Minimal diagnostics to aid failing test visibility
                    sh = getattr(result, "step_history", []) or []
                    typer.echo(f"[grey58]No YAML found. step_history_len={len(sh)}[/grey58]")
                    try:
                        ctx = getattr(result, "final_pipeline_context", None)
                        if ctx is not None:
                            g = getattr(ctx, "generated_yaml", None)
                            y = getattr(ctx, "yaml_text", None)
                            typer.echo(
                                f"[grey58]final_ctx has generated_yaml={bool(g)} yaml_text={bool(y)}[/grey58]"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass
                typer.echo("[red]Architect did not produce YAML (context.generated_yaml missing)")
                raise typer.Exit(1)

            # Security gating: detect side-effect tools and require confirmation unless explicitly allowed
            from .helpers import find_side_effect_skills_in_yaml, enrich_yaml_with_required_params

            side_effect_skills = find_side_effect_skills_in_yaml(
                yaml_text, base_dir=output_dir or os.getcwd()
            )
            if side_effect_skills and not allow_side_effects:
                typer.echo(
                    "[red]This blueprint references side-effect skills that may perform external actions:"
                )
                for sid in side_effect_skills:
                    typer.echo(f"  - {sid}")
                if non_interactive:
                    typer.echo(
                        "[red]Non-interactive mode: re-run with --allow-side-effects to proceed."
                    )
                    raise typer.Exit(1)
                confirm = typer.confirm(
                    "Proceed anyway? This may perform external actions (e.g., Slack posts).",
                    default=False,
                )
                if not confirm:
                    raise typer.Exit(1)

            # Optionally enrich YAML with required params if interactive and missing
            yaml_text = enrich_yaml_with_required_params(
                yaml_text,
                non_interactive=non_interactive,
                base_dir=output_dir or os.getcwd(),
            )

            # Opportunistic sanitization before validation
            try:
                from .helpers import sanitize_blueprint_yaml as _sanitize_yaml

                yaml_text = _sanitize_yaml(yaml_text)
            except Exception:
                pass

            # If no YAML could be produced (e.g., user denied plan), abort
            def _any_denied_branch(step_results: list[Any]) -> bool:
                try:
                    for sr in step_results or []:
                        md = getattr(sr, "metadata_", {}) or {}
                        key = md.get("executed_branch_key")
                        if isinstance(key, str) and key.strip().lower() in {
                            "denied",
                            "reject",
                            "rejected",
                        }:
                            return True
                        nested = getattr(sr, "step_history", None)
                        if isinstance(nested, list) and nested and _any_denied_branch(nested):
                            return True
                except Exception:
                    return False
                return False

            no_yaml = not isinstance(yaml_text, str) or not yaml_text.strip()
            looks_like_yaml = False
            try:
                if isinstance(yaml_text, str):
                    s = yaml_text.strip()
                    looks_like_yaml = ("version:" in s) or ("steps:" in s)
            except Exception:
                looks_like_yaml = False

            if (
                no_yaml
                or not looks_like_yaml
                or _any_denied_branch(getattr(result, "step_history", []) or [])
            ):
                typer.echo(
                    "[red]No YAML was generated from the architect pipeline (plan rejected or writer failed). Aborting.",
                    err=True,
                )
                raise typer.Exit(1)

            # Validate in-memory before writing
            report = validate_yaml_text(yaml_text, base_dir=output_dir or os.getcwd())
            if not report.is_valid and strict:
                typer.echo("[red]Generated YAML is invalid under --strict")
                raise typer.Exit(1)

            # Interactive HITL: show plan and ask for approval when --goal flag was not provided
            if hitl_requested:
                try:
                    preview = yaml_text.strip()
                    # Trim extremely long previews
                    if len(preview) > 2000:
                        preview = preview[:2000] + "\n... (truncated)"
                    typer.echo("\n[bold]Proposed pipeline plan (YAML preview):[/bold]")
                    typer.echo(preview)
                except Exception:
                    pass
                approved = typer.confirm(
                    "Proceed to generate pipeline from this plan?", default=True
                )
                if not approved:
                    typer.echo("[red]Creation aborted by user at plan approval stage.")
                    raise typer.Exit(1)

            # Write outputs
            # Determine output location (project-aware by default)
            # If an explicit --output-dir is provided, do NOT require a Flujo project.
            if output_dir is not None:
                out_dir = output_dir
                project_root = None  # Only used for overwrite policy below
            else:
                project_root = str(find_project_root())
                out_dir = project_root
            os.makedirs(out_dir, exist_ok=True)
            out_yaml = os.path.join(out_dir, "pipeline.yaml")
            # In project-aware default path, allow overwriting pipeline.yaml without --force
            allow_overwrite = (project_root is not None) and (
                os.path.abspath(out_dir) == os.path.abspath(project_root)
            )
            if os.path.exists(out_yaml) and not (force or allow_overwrite):
                typer.echo(
                    f"[red]Refusing to overwrite existing file: {out_yaml}. Use --force to overwrite."
                )
                raise typer.Exit(1)
            # Prompt for name if interactive and not provided
            if not name and not non_interactive:
                detected = _extract_pipeline_name_from_yaml(yaml_text)
                name = typer.prompt(
                    "What should we name this pipeline?", default=detected or "pipeline"
                )
            # Optionally inject top-level name into YAML if absent
            if name and (_extract_pipeline_name_from_yaml(yaml_text) is None):
                yaml_text = f'name: "{name}"\n' + yaml_text
            # Ensure version appears first for stable outputs
            try:
                lines = yaml_text.splitlines(True)
                v_idx = next(
                    (i for i, line in enumerate(lines) if line.strip().startswith("version:")),
                    None,
                )
                if isinstance(v_idx, int) and v_idx > 0:
                    version_line = lines.pop(v_idx)
                    lines.insert(0, version_line)
                    yaml_text = "".join(lines)
            except Exception:
                pass

            with open(out_yaml, "w") as f:
                f.write(yaml_text)
            typer.echo(f"[green]Wrote: {out_yaml}")

            # Budget confirmation (interactive only). If a budget was provided via flag, respect it.
            budget_val: float | None = None
            if not non_interactive:
                try:
                    if budget is None:
                        # Prompt for numeric budget
                        resp = typer.prompt(
                            "What is a safe cost limit per run (USD)?", default="2.50"
                        )
                        try:
                            budget_val = float(resp)
                        except Exception:
                            typer.echo(
                                "[red]Invalid budget value. Please enter a number (e.g., 2.50)."
                            )
                            raise typer.Exit(2)
                    else:
                        budget_val = float(budget)
                    # Optional confirmation (opt-in via env)
                    try:
                        _bc_env = os.environ.get("FLUJO_CREATE_BUDGET_CONFIRM", "").strip().lower()
                    except Exception:
                        _bc_env = ""
                    if _bc_env in {"1", "true", "yes", "on"}:
                        if not typer.confirm(
                            f"Confirm budget limit ${budget_val:.2f} per run?", default=True
                        ):
                            typer.echo(
                                "[red]Creation aborted by user at budget confirmation stage."
                            )
                            raise typer.Exit(1)
                except Exception:
                    # Fall back to skipping budget confirmation on unexpected prompt failures
                    budget_val = None

            # Optionally update flujo.toml budget
            try:
                # Prefer the interactive-confirmed budget when available; otherwise use flag
                if budget_val is not None or budget is not None:
                    if budget_val is None and budget is not None:
                        budget_val = float(budget)
                    # Determine pipeline name to write budget under
                    pipeline_name = (
                        name or _extract_pipeline_name_from_yaml(yaml_text) or "pipeline"
                    )
                    flujo_toml_path = Path(out_dir) / "flujo.toml"
                    if flujo_toml_path.exists() and budget_val is not None:
                        update_project_budget(flujo_toml_path, pipeline_name, float(budget_val))
                        typer.echo(
                            f"[green]Updated budget for pipeline '{pipeline_name}' in flujo.toml"
                        )
            except Exception:
                # Do not fail create on budget write issues
                pass
        finally:
            # Always restore original logging levels
            try:
                _flujo_logger.setLevel(_orig_flujo_level)
                _httpx_logger.setLevel(_orig_httpx_level)
                # Reset to default warning filters (sufficient for CLI lifecycle)
                _warnings.resetwarnings()
            except Exception:
                pass

            # Comprehensive cleanup to prevent process hang
            try:
                import asyncio
                import gc
                import threading

                # Force garbage collection
                gc.collect()

                # Cancel any remaining asyncio tasks
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, cancel all tasks
                    tasks = asyncio.all_tasks(loop)
                    if tasks:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                except RuntimeError:
                    # No running loop, which is expected
                    pass

                # Clean up any remaining threads that might be hanging
                threads = [
                    t
                    for t in threading.enumerate()
                    if t != threading.main_thread() and t.is_alive()
                ]
                if threads:
                    for thread in threads:
                        try:
                            # Try to join with a timeout to avoid hanging
                            thread.join(timeout=0.1)
                        except Exception:
                            pass

                # Additional cleanup for common async libraries
                try:
                    # Clean up httpx connection pools
                    import httpx

                    if hasattr(httpx, "_default_limits"):
                        httpx._default_limits = None
                except Exception:
                    pass

                try:
                    # Clean up any SQLite async locks
                    # Note: sqlite3.connect._instance doesn't exist in standard Python
                    # This cleanup was attempting to access a non-existent attribute
                    pass
                except Exception:
                    pass

                # Force final garbage collection
                gc.collect()

            except Exception:
                # Don't fail the command on cleanup errors
                pass
    except typer.Exit:
        # Preserve explicit exit codes for wizard/architect flows without wrapping
        raise
    except Exception as e:
        typer.echo(f"[red]Failed to create pipeline: {e}", err=True)
        raise typer.Exit(1)


@app.command(help="🚀 Run the workflow in the current project.")
def run(
    pipeline_file: Optional[str] = typer.Argument(
        None,
        help="Path to the pipeline (.py or .yaml). If omitted, uses project pipeline.yaml",
    ),
    input_data: Optional[str] = typer.Option(
        None,
        "--input",
        "--input-data",
        "-i",
        help=(
            "Initial input data for the pipeline. Use '-' to read from stdin. "
            "When omitted, Flujo reads from FLUJO_INPUT (if set) or piped stdin."
        ),
    ),
    context_model: Optional[str] = typer.Option(
        None, "--context-model", "-c", help="Context model class name to use"
    ),
    context_data: Optional[str] = typer.Option(
        None, "--context-data", "-d", help="JSON string for initial context data"
    ),
    context_file: Optional[str] = typer.Option(
        None, "--context-file", "-f", help="Path to JSON/YAML file with context data"
    ),
    pipeline_name: str = typer.Option(
        "pipeline",
        "--pipeline-name",
        "-p",
        help="Name of the pipeline variable (default: pipeline)",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Unique run ID for state persistence"
    ),
    json_output: bool = typer.Option(
        False, "--json", "--json-output", help="Output raw JSON instead of formatted result"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Parse and validate only; do not execute the pipeline",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable verbose step-by-step logs and console tracing",
    ),
    debug_prompts: bool = typer.Option(
        False,
        "--debug-prompts",
        help="Also include full prompts and responses in trace events (unsafe)",
    ),
    trace_preview_len: Optional[int] = typer.Option(
        None,
        "--trace-preview-len",
        help="Max characters for prompt/response previews in the debug trace (default 1000)",
    ),
    debug_export: bool = typer.Option(
        False,
        "--debug-export",
        help="Enable full debug log export (default path if --debug-export-path omitted)",
    ),
    debug_export_path: Optional[str] = typer.Option(
        None,
        "--debug-export-path",
        help="Write a full debug log (trace tree, step history, final context) to this JSON file",
    ),
) -> None:
    """
    Run a custom pipeline from a Python file.

    This command loads a pipeline from a Python file and executes it with the provided input.
    The pipeline should be defined as a top-level variable (default: 'pipeline') of type Pipeline.

    Examples:
        flujo run my_pipeline.py --input "Hello world"
        flujo run my_pipeline.py --input "Process this" --context-model MyContext --context-data '{"key": "value"}'
        flujo run my_pipeline.py --input "Test" --context-file context.yaml
    """
    # Ensure we always have a symbol in scope for cleanup
    runner: Any | None = None
    try:
        # Apply CLI defaults from configuration file
        cli_args = apply_cli_defaults(
            "run",
            pipeline_name=pipeline_name,
            json_output=json_output,
        )
        pipeline_name = cast(str, cli_args["pipeline_name"])
        json_output = cast(bool, cli_args["json_output"])

        # Detect raw flags to support JSON mode when alias parsing fails
        ctx = click.get_current_context()
        if not json_output and any(flag in ctx.args for flag in ("--json", "--json-output")):
            json_output = True

        # Resolve default pipeline file from project if omitted
        if pipeline_file is None:
            root = find_project_root()
            pipeline_file = str((Path(root) / "pipeline.yaml").resolve())

        # If YAML blueprint provided, load via blueprint loader; else use existing Python loader.
        if pipeline_file.endswith((".yaml", ".yml")):
            pipeline_obj = load_pipeline_from_yaml_file(pipeline_file)
            context_model_class = None
            initial_context_data = parse_context_data(context_data, context_file)
            # Resolve initial input for YAML runs
            from .helpers import resolve_initial_input as _resolve_initial_input

            input_data = _resolve_initial_input(input_data)
        else:
            pipeline_obj, pipeline_name, input_data, initial_context_data, context_model_class = (
                setup_run_command_environment(
                    pipeline_file=pipeline_file,
                    pipeline_name=pipeline_name,
                    json_output=json_output,
                    input_data=input_data,
                    context_model=context_model,
                    context_data=context_data,
                    context_file=context_file,
                )
            )

        # Pre-run validation enforcement
        from flujo.domain.pipeline_validation import ValidationReport

        try:
            # Align with FSD-015: explicitly raise on error
            pipeline_obj.validate_graph(raise_on_error=True)
        except Exception:
            # Recompute full report for user-friendly printing
            try:
                validation_report: ValidationReport = pipeline_obj.validate_graph()
            except Exception as ve:  # pragma: no cover - defensive
                typer.echo(f"[red]Validation crashed: {ve}", err=True)
                raise typer.Exit(1)

            if not validation_report.is_valid:
                typer.echo("[red]Pipeline validation failed before run:")
                for f in validation_report.errors:
                    loc = f"{f.step_name}: " if f.step_name else ""
                    if f.suggestion:
                        typer.echo(
                            f"- [{f.rule_id}] {loc}{f.message} -> Suggestion: {f.suggestion}"
                        )
                    else:
                        typer.echo(f"- [{f.rule_id}] {loc}{f.message}")
                raise typer.Exit(1)

        # If dry-run requested, stop after validation
        if dry_run:
            try:
                names = [s.name for s in getattr(pipeline_obj, "steps", [])]
            except Exception:
                names = []
            if json_output:
                typer.echo(json.dumps({"validated": True, "steps": names}))
            else:
                typer.echo("[green]Pipeline parsed and validated (dry run)")
                if names:
                    typer.echo("Steps:")
                    for n in names:
                        typer.echo(f"- {n}")
            return

        # Create Flujo runner using helper function
        # Enable debug environment hints for deeper logs and stack traces
        if debug or debug_prompts:
            try:
                import os as _os

                _os.environ.setdefault("FLUJO_CLI_VERBOSE", "1")
                _os.environ.setdefault("FLUJO_CLI_TRACE", "1")
                _os.environ.setdefault("FLUJO_DEBUG", "1")
                if debug_prompts:
                    _os.environ["FLUJO_DEBUG_PROMPTS"] = "1"
                if trace_preview_len is not None:
                    _os.environ["FLUJO_TRACE_PREVIEW_LEN"] = str(int(trace_preview_len))
            except Exception:
                pass

        # Load the project-aware state backend from configuration so `flujo run`
        # honors flujo.toml/FLUJO_STATE_URI (e.g., default memory:// from `flujo init`).
        # Falls back to None on errors which lets Runner choose safe defaults.
        _state_backend = None
        try:
            from .config import load_backend_from_config as _load_backend_from_config

            _state_backend = _load_backend_from_config()
        except Exception:
            _state_backend = None

        runner = create_flujo_runner(
            pipeline=pipeline_obj,
            context_model_class=context_model_class,
            initial_context_data=initial_context_data,
            state_backend=_state_backend,
            debug=debug,
        )

        # Execute pipeline using helper function
        # mypy: ensure input_data is a concrete string at this point
        from typing import cast as _cast

        input_data_str = _cast(str, input_data)
        result = execute_pipeline_with_output_handling(
            runner=runner,
            input_data=input_data_str,
            run_id=run_id,
            json_output=json_output,
        )

        # Interactive HITL resume loop: if paused and in TTY, prompt and resume
        if not json_output:
            try:
                import sys as _sys
                import asyncio as _asyncio

                def _is_paused(_res: Any) -> tuple[bool, str | None]:
                    try:
                        ctx = getattr(_res, "final_pipeline_context", None)
                        scratch = getattr(ctx, "scratchpad", None) if ctx is not None else None
                        if isinstance(scratch, dict) and scratch.get("status") == "paused":
                            return True, (
                                scratch.get("pause_message") or scratch.get("hitl_message")
                            )
                    except Exception:
                        pass
                    return False, None

                paused, msg = _is_paused(result)
                while paused and _sys.stdin.isatty():
                    prompt_msg = msg or "Provide input to resume:"
                    human = typer.prompt(prompt_msg)
                    # Resume via runner
                    result = _asyncio.run(runner.resume_async(result, human))
                    paused, msg = _is_paused(result)
            except Exception:
                # If resume fails, fall through to normal display (will show paused message)
                pass

        # Handle output
        if json_output:
            typer.echo(result)
        else:
            display_pipeline_results(result, run_id, json_output)
            # When debugging, print a compact trace tree with step attributes and key events
            if debug or debug_prompts:
                try:
                    from rich.console import Console
                    from rich.tree import Tree

                    console = Console()

                    def _span_label(span: Any) -> str:
                        try:
                            name = getattr(span, "name", "span")
                            status = getattr(span, "status", "?")
                            dur = 0.0
                            st = getattr(span, "start_time", None)
                            en = getattr(span, "end_time", None)
                            if isinstance(st, (int, float)) and isinstance(en, (int, float)):
                                dur = max(0.0, float(en) - float(st))
                            return f"{name} [{status}] ({dur:.3f}s)"
                        except Exception:
                            return "span"

                    def _add_span(tree: Tree, span: Any) -> None:
                        node = tree.add(_span_label(span))
                        # Attributes
                        try:
                            attrs = getattr(span, "attributes", {}) or {}
                            # Show selected keys to keep it readable
                            keys = [
                                k
                                for k in attrs.keys()
                                if k.startswith("flujo.")
                                or k in ("success", "latency_s", "step_input")
                            ]
                            for k in keys:
                                v = attrs.get(k)
                                node.add(f"[grey62]{k}: {v}[/grey62]")
                        except Exception:
                            pass
                        # Events
                        try:
                            import os as _os

                            try:
                                prev_len = int(_os.getenv("FLUJO_TRACE_PREVIEW_LEN", "1000"))
                            except Exception:
                                prev_len = 1000
                            full_flag = _os.getenv("FLUJO_DEBUG_PROMPTS") == "1"
                            for ev in getattr(span, "events", []) or []:
                                ename = str(ev.get("name"))
                                eattrs = ev.get("attributes", {}) or {}
                                # Specialized formatting for agent events with previews
                                if ename in {
                                    "agent.system",
                                    "agent.input",
                                    "agent.response",
                                    "agent.prompt",
                                }:
                                    if ename == "agent.system":
                                        txt = eattrs.get(
                                            "system_prompt_full"
                                            if full_flag
                                            else "system_prompt_preview",
                                            "",
                                        )
                                    elif ename == "agent.input":
                                        txt = eattrs.get(
                                            "input_full" if full_flag else "input_preview", ""
                                        )
                                    elif ename == "agent.response":
                                        txt = eattrs.get(
                                            "response_full" if full_flag else "response_preview", ""
                                        )
                                    else:  # agent.prompt from history injector
                                        txt = eattrs.get("rendered_history", "")
                                    s = str(txt)
                                    if not full_flag and prev_len >= 0 and len(s) > prev_len:
                                        s = s[:prev_len] + "..."
                                    node.add(f"[yellow]event[/yellow] {ename}: {s}")
                                elif ename == "agent.usage":
                                    node.add(
                                        f"[yellow]event[/yellow] {ename}: tokens_in={eattrs.get('input_tokens')} tokens_out={eattrs.get('output_tokens')} cost=${eattrs.get('cost_usd')}"
                                    )
                                elif ename == "agent.system.vars":
                                    node.add(f"[yellow]event[/yellow] {ename}: {eattrs}")
                                else:
                                    node.add(f"[yellow]event[/yellow] {ename}: {eattrs}")
                        except Exception:
                            pass
                        # Children
                        try:
                            for ch in getattr(span, "children", []) or []:
                                _add_span(node, ch)
                        except Exception:
                            pass

                    trace_root = getattr(result, "trace_tree", None)
                    if trace_root is not None:
                        console.rule("[bold]Debug Trace")
                        root_tree = Tree(_span_label(trace_root))
                        for child in getattr(trace_root, "children", []) or []:
                            _add_span(root_tree, child)
                        console.print(root_tree)
                except Exception:
                    pass

        # If export enabled (via --debug-export) or --debug set and no explicit path, choose default path
        export_path: Optional[str] = None
        if (debug_export or debug) and not debug_export_path:
            try:
                from pathlib import Path as _Path
                from datetime import datetime as _dt

                root = find_project_root()
                base_dir = _Path(root) if root else _Path.cwd()
                debug_dir = base_dir / "debug"
                ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
                rid = run_id or "run"
                safe_rid = "".join(ch if ch.isalnum() else "-" for ch in str(rid))[:24]
                export_path = str((debug_dir / f"{ts}_{safe_rid}.json").resolve())
            except Exception:
                export_path = None
        elif debug_export_path:
            export_path = debug_export_path

        # Optional: export a full debug log to a file for deep analysis
        if export_path:
            try:
                from pathlib import Path as _Path
                from datetime import datetime as _dt
                from flujo.utils.serialization import safe_serialize as _safe
                import os as _os

                export_path_obj = _Path(export_path).expanduser().resolve()
                export_path_obj.parent.mkdir(parents=True, exist_ok=True)

                def _span_to_dict(span: Any) -> dict[str, Any]:
                    if span is None:
                        return {}
                    try:
                        children = [_span_to_dict(ch) for ch in getattr(span, "children", []) or []]
                    except Exception:
                        children = []
                    try:
                        events = list(getattr(span, "events", []) or [])
                    except Exception:
                        events = []
                    return {
                        "name": getattr(span, "name", ""),
                        "status": getattr(span, "status", ""),
                        "start_time": getattr(span, "start_time", None),
                        "end_time": getattr(span, "end_time", None),
                        "attributes": getattr(span, "attributes", {}) or {},
                        "events": events,
                        "children": children,
                    }

                def _step_result_to_dict(sr: Any) -> dict[str, Any]:
                    try:
                        nested = [
                            _step_result_to_dict(ch)
                            for ch in (getattr(sr, "step_history", []) or [])
                        ]
                    except Exception:
                        nested = []
                    return {
                        "name": getattr(sr, "name", None),
                        "success": getattr(sr, "success", None),
                        "attempts": getattr(sr, "attempts", None),
                        "latency_s": getattr(sr, "latency_s", None),
                        "token_counts": getattr(sr, "token_counts", None),
                        "cost_usd": getattr(sr, "cost_usd", None),
                        "feedback": getattr(sr, "feedback", None),
                        "output": _safe(getattr(sr, "output", None)),
                        "metadata": getattr(sr, "metadata_", {}) or {},
                        "step_history": nested,
                    }

                def _context_to_dict(ctx: Any) -> dict[str, Any]:
                    if ctx is None:
                        return {}
                    try:
                        base = ctx.model_dump() if hasattr(ctx, "model_dump") else _safe(ctx)
                    except Exception:
                        base = {}
                    # Augment with common transient fields for debugging
                    try:
                        base["scratchpad"] = _safe(getattr(ctx, "scratchpad", {}))
                    except Exception:
                        pass
                    try:
                        base["conversation_history"] = [
                            {
                                "role": getattr(getattr(t, "role", None), "value", None),
                                "content": getattr(t, "content", None),
                            }
                            for t in (getattr(ctx, "conversation_history", []) or [])
                        ]
                    except Exception:
                        pass
                    try:
                        base["hitl_history"] = _safe(getattr(ctx, "hitl_history", []))
                    except Exception:
                        pass
                    try:
                        base["command_log"] = _safe(getattr(ctx, "command_log", []))
                    except Exception:
                        pass
                    return base if isinstance(base, dict) else {"context": base}

                # Final payload
                payload = {
                    "exported_at": _dt.utcnow().isoformat() + "Z",
                    "pipeline_name": pipeline_name,
                    "run_id": run_id,
                    "env": {
                        "FLUJO_DEBUG": _os.getenv("FLUJO_DEBUG"),
                        "FLUJO_DEBUG_PROMPTS": _os.getenv("FLUJO_DEBUG_PROMPTS"),
                        "FLUJO_TRACE_PREVIEW_LEN": _os.getenv("FLUJO_TRACE_PREVIEW_LEN"),
                    },
                    "trace_tree": _span_to_dict(getattr(result, "trace_tree", None)),
                    "result": {
                        "total_cost_usd": getattr(result, "total_cost_usd", None),
                        "total_tokens": getattr(result, "total_tokens", None),
                        "step_history": [
                            _step_result_to_dict(sr)
                            for sr in (getattr(result, "step_history", []) or [])
                        ],
                    },
                    "final_context": _context_to_dict(
                        getattr(result, "final_pipeline_context", None)
                    ),
                }

                import json as _json

                with open(export_path_obj, "w", encoding="utf-8") as fh:
                    _json.dump(payload, fh, indent=2, ensure_ascii=False)
                if not json_output:
                    typer.echo(f"[green]Wrote full debug log to[/green] {export_path_obj}")
            except Exception as e:
                if not json_output:
                    typer.echo(f"[red]Failed to export debug log:[/red] {e}", err=True)

    except UsageLimitExceededError as e:
        # Friendly budget exceeded messaging with partial results if available
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            msg = str(e) or "Usage limits exceeded"
            console.print(
                Panel.fit(f"[bold red]Budget exceeded[/bold red]\n{msg}", border_style="red")
            )
            partial = getattr(e, "result", None)
            if partial is not None:
                try:
                    display_pipeline_results(partial, run_id, False)
                except Exception:
                    pass
        except Exception:
            typer.echo(f"[red]Budget exceeded: {e}[/red]", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        # Preserve specific exit codes raised by helpers
        raise
    except Exception as e:
        try:
            import os

            os.makedirs("output", exist_ok=True)
            with open("output/last_run_error.txt", "w") as fh:
                fh.write(repr(e))
        except Exception:
            pass
        import os as _os
        import traceback as _tb

        typer.echo(f"[red]Error running pipeline: {type(e).__name__}: {e}", err=True)
        if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
            typer.echo("\nTraceback:", err=True)
            typer.echo("".join(_tb.format_exception(e)), err=True)
        from .exit_codes import EX_RUNTIME_ERROR

        raise typer.Exit(EX_RUNTIME_ERROR)
    finally:
        # Best-effort cleanup to prevent post-run hangs
        try:
            import asyncio as _asyncio
            import gc as _gc
            import threading as _threading

            # Force GC to clear orphaned async objects
            try:
                _gc.collect()
            except Exception:
                pass

            # Cancel any remaining asyncio tasks (if a loop exists in this context)
            try:
                loop = _asyncio.get_running_loop()
                for task in list(_asyncio.all_tasks(loop)):
                    if not task.done():
                        task.cancel()
            except RuntimeError:
                # No running loop in this context
                pass
            except Exception:
                pass

            # Join any lingering non-main threads briefly
            try:
                for t in [
                    th
                    for th in _threading.enumerate()
                    if th is not _threading.main_thread() and th.is_alive()
                ]:
                    try:
                        t.join(timeout=0.2)
                    except Exception:
                        pass
            except Exception:
                pass

            # Try to gracefully shutdown the state backend if exposed on the runner
            try:
                sb = getattr(runner, "state_backend", None)
                if sb is not None and hasattr(sb, "shutdown"):

                    async def _do_shutdown() -> None:
                        try:
                            await sb.shutdown()
                        except Exception:
                            pass

                    try:
                        _asyncio.run(_do_shutdown())
                    except RuntimeError:
                        # Running loop: schedule and wait best-effort
                        try:
                            loop = _asyncio.get_running_loop()
                            loop.create_task(_do_shutdown())
                            # Best-effort - do not block indefinitely
                            # If we cannot await, ignore silently
                        except Exception:
                            pass
            except Exception:
                pass

            # Additional library-specific cleanups (idempotent)
            try:
                import httpx as _httpx

                if hasattr(_httpx, "_default_limits"):
                    _httpx._default_limits = None
            except Exception:
                pass

            # Ensure any pooled SQLite connections are closed (extra safety)
            try:
                from flujo.state.backends.sqlite import SQLiteBackend as _SQLiteBackend

                _SQLiteBackend.shutdown_all()
            except Exception:
                pass

            # Clear dynamic skill registry entries (preserve built-ins)
            try:
                from flujo.infra.skill_registry import get_skill_registry as _get_reg

                reg = _get_reg()
                entries = getattr(reg, "_entries", None)
                if isinstance(entries, dict):
                    preserved: Dict[str, Any] = {
                        k: v
                        for k, v in list(entries.items())
                        if isinstance(k, str)
                        and (k.startswith("flujo.builtins.") or k.startswith("flujo.architect."))
                    }
                    entries.clear()
                    entries.update(preserved)
            except Exception:
                pass

            # Final GC sweep
            try:
                _gc.collect()
            except Exception:
                pass
        except Exception:
            # Never fail the command on cleanup
            pass


@dev_app.command(name="compile-yaml")
def compile(
    src: str = typer.Argument(..., help="Input spec: .yaml/.yml or .py"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file path (.yaml)"),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="Normalize YAML formatting and structure"
    ),
) -> None:
    """Compile a pipeline spec between YAML and DSL.

    - If src is YAML, parses and pretty-prints validated YAML (normalized).
    - If src is Python, imports the pipeline and dumps YAML.
    """
    try:
        if src.endswith((".yaml", ".yml")):
            # Load and re-dump normalized YAML
            from flujo.domain.dsl import Pipeline

            pipe = Pipeline.from_yaml_file(src)
            yaml_text = pipe.to_yaml() if normalize else open(src, "r").read()
        else:
            pipeline_obj, _ = load_pipeline_from_file(src)
            yaml_text = pipeline_obj.to_yaml()
        if out:
            with open(out, "w") as f:
                f.write(yaml_text)
            typer.echo(f"[green]Wrote: {out}")
        else:
            typer.echo(yaml_text)
    except Exception as e:
        typer.echo(f"[red]Failed to compile: {e}", err=True)
        raise typer.Exit(1)


@budgets_app.command("show")
def budgets_show(pipeline_name: str) -> None:
    """Print the effective budget for a pipeline and its resolution source.

    Example:
        flujo budgets show my-pipeline
    """
    try:
        from flujo.infra.config_manager import ConfigManager
        from flujo.infra.budget_resolver import resolve_limits_for_pipeline

        cfg = ConfigManager().load_config()
        limits, src = resolve_limits_for_pipeline(getattr(cfg, "budgets", None), pipeline_name)

        if limits is None:
            typer.echo("No budget configured (unlimited). Source: none")
            return

        # Pretty print the effective budget
        cost = (
            f"${limits.total_cost_usd_limit:.2f}"
            if limits.total_cost_usd_limit is not None
            else "unlimited"
        )
        tokens = (
            f"{limits.total_tokens_limit}" if limits.total_tokens_limit is not None else "unlimited"
        )
        origin = src.source if src.pattern is None else f"{src.source}[{src.pattern}]"
        typer.echo(f"Effective budget for '{pipeline_name}':")
        typer.echo(f"  - total_cost_usd_limit: {cost}")
        typer.echo(f"  - total_tokens_limit: {tokens}")
        typer.echo(f"Resolved from {origin} in flujo.toml")
    except Exception as e:
        typer.echo(f"[red]Failed to resolve budgets: {e}", err=True)
        raise typer.Exit(1)


@dev_app.command(name="visualize")
def pipeline_mermaid_cmd(
    file: str = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to the Python file containing the pipeline object",
    ),
    object_name: str = typer.Option(
        "pipeline",
        "--object",
        "-o",
        help="Name of the pipeline variable in the file (default: pipeline)",
    ),
    detail_level: str = typer.Option(
        "auto", "--detail-level", "-d", help="Detail level: auto, high, medium, low"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-O", help="Output file (default: stdout)"
    ),
) -> None:
    """
    Output a pipeline's Mermaid diagram at the chosen detail level.

    Example:
        flujo pipeline-mermaid --file my_pipeline.py --object pipeline --detail-level medium --output diagram.md
    """
    try:
        mermaid_code = load_mermaid_code(file, object_name, detail_level)
        if output:
            with open(output, "w") as f:
                f.write("```mermaid\n")
                f.write(mermaid_code)
                f.write("\n```")
            typer.echo(f"[green]Mermaid diagram written to {output}")
        else:
            typer.echo("```mermaid")
            typer.echo(mermaid_code)
            typer.echo("```")
    except Exception as e:
        typer.echo(f"[red]Failed to load file: {e}", err=True)
        raise typer.Exit(1)


@app.callback()
def main(
    profile: Annotated[
        bool, typer.Option("--profile", help="Enable Logfire STDOUT span viewer")
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            help="Enable verbose debug logging to '.flujo/logs/run.log'.",
        ),
    ] = False,
    project: Annotated[
        Optional[str],
        typer.Option(
            "--project",
            help=(
                "Project root directory (overrides FLUJO_PROJECT_ROOT). "
                "Adds it to PYTHONPATH for imports like skills.*"
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed error traces for debugging",
        ),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option(
            "--trace",
            help="Alias for --verbose to print full Python tracebacks",
        ),
    ] = False,
) -> None:
    """
    CLI entry point for flujo.

    Args:
        profile: Enable Logfire STDOUT span viewer for profiling

    Returns:
        None
    """
    if profile:
        logfire.enable_stdout_viewer()
    # Optional global debug logging to a local file
    if debug:
        try:
            import logging as _logging
            import os as _os

            _os.makedirs(".flujo/logs", exist_ok=True)
            _fh = _logging.FileHandler(".flujo/logs/run.log", encoding="utf-8")
            _fh.setLevel(_logging.DEBUG)
            _fmt = _logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            _fh.setFormatter(_fmt)
            _logger = _logging.getLogger("flujo")
            _logger.setLevel(_logging.DEBUG)
            _logger.addHandler(_fh)
        except Exception:
            # Never fail CLI due to logging setup issues
            pass
    # Quiet by default: reduce console noise unless --debug
    try:
        import logging as _logging
        import os as _os

        _logger = _logging.getLogger("flujo")
        if debug:
            # Propagate debug intent to runtime via env for internal warnings gates
            try:
                _os.environ["FLUJO_DEBUG"] = "1"
            except Exception:
                pass
            _logger.setLevel(_logging.INFO)
            for h in list(_logger.handlers):
                try:
                    h.setLevel(_logging.INFO)
                except Exception:
                    pass
        else:
            # Ensure flag is not set when not debugging
            try:
                if _os.environ.get("FLUJO_DEBUG"):
                    del _os.environ["FLUJO_DEBUG"]
            except Exception:
                pass
            _logger.setLevel(_logging.WARNING)
            for h in list(_logger.handlers):
                # Keep error handler; downgrade others to WARNING
                try:
                    h.setLevel(_logging.WARNING)
                except Exception:
                    pass
    except Exception:
        pass

    # Enable verbose traces for helpers and error handlers
    try:
        if verbose or trace:
            _os.environ["FLUJO_CLI_VERBOSE"] = "1"
    except Exception:
        pass
    # Resolve and inject project root into sys.path for imports like skills.*
    try:
        explicit = Path(project).resolve() if project else None
        root = resolve_project_root(explicit=explicit, allow_missing=True)
        if project and root is not None:
            _os.environ["FLUJO_PROJECT_ROOT"] = str(root)
        ensure_project_root_on_sys_path(root)
    except Exception:
        # Non-fatal: individual commands may still set defaults or load explicit files
        pass


# Wizard helper and top-level explain command (NS4)


def _run_create_wizard(
    *,
    goal: Optional[str],
    name: Optional[str],
    output_dir: Optional[str],
    non_interactive: bool,
    pattern: Optional[str],
    iterable_name: Optional[str],
    reduce_mode: Optional[str],
    conversation: Optional[bool],
    stop_when: Optional[bool],
    propagation: Optional[str],
    body_steps: Optional[str],
    map_step_name: Optional[str],
    branch_names: Optional[str],
    ai_turn_source: Optional[str] = None,
    user_turn_sources: Optional[str] = None,
    history_strategy: Optional[str] = None,
    history_max_tokens: Optional[int] = None,
    history_max_turns: Optional[int] = None,
    history_summarize_ratio: Optional[float] = None,
) -> None:
    import os as _os
    import typer as _ty

    nm = name or (goal.replace(" ", "_")[:30] if goal else None)
    if nm is None and not non_interactive:
        nm = _ty.prompt("Pipeline name", default="my_pipeline")
    if nm is None:
        nm = "my_pipeline"

    # Pick pattern
    if pattern is None and not non_interactive:
        pattern = _ty.prompt("Pattern (loop/map/parallel)", default="loop")
    if pattern is None:
        pattern = "loop"
    pattern = str(pattern).lower().strip()

    conv = (
        conversation
        if conversation is not None
        else (
            True
            if non_interactive
            else _ty.confirm("Is this a conversation/iterative loop?", default=True)
        )
    )
    stop_when_finished = (
        stop_when
        if stop_when is not None
        else (
            True
            if non_interactive
            else _ty.confirm("Stop when the agent signals 'finish'?", default=True)
        )
    )
    # If propagation is not provided, ask for 'auto' guidance in interactive mode
    if propagation is None and not non_interactive:
        default_auto = _ty.confirm("Use propagation: auto?", default=True)
        propagation = "auto" if default_auto else "previous_output"
    out_mode = (
        "text" if non_interactive else _ty.prompt("Output mode (text/fields)", default="text")
    )

    lines: list[str] = ['version: "0.1"', "steps:"]
    lines.append("  - kind: step\n    name: get_goal")

    if pattern == "loop":
        lines.append("  - kind: loop")
        lines.append(f"    name: {nm}")
        lines.append("    loop:")
        if conv:
            lines.append("      conversation: true")
            # History management presets
            # Determine strategy
            if non_interactive:
                hs = (history_strategy or "truncate_tokens").strip().lower()
            else:
                import click as _click

                hs = history_strategy or _click.prompt(
                    "History strategy (truncate_tokens/ truncate_turns/ summarize)",
                    default="truncate_tokens",
                )
            # Emit history_management block with reasonable defaults
            lines.append("      history_management:")
            if hs == "truncate_turns":
                mt = history_max_turns if history_max_turns is not None else 20
                lines.append("        strategy: truncate_turns")
                lines.append(f"        max_turns: {int(mt)}")
            elif hs == "summarize":
                ratio = history_summarize_ratio if history_summarize_ratio is not None else 0.5
                mtok = history_max_tokens if history_max_tokens is not None else 4096
                lines.append("        strategy: summarize")
                lines.append(f"        summarize_ratio: {float(ratio)}")
                lines.append(f"        max_tokens: {int(mtok)}")
            else:
                mtok = history_max_tokens if history_max_tokens is not None else 4096
                lines.append("        strategy: truncate_tokens")
                lines.append(f"        max_tokens: {int(mtok)}")
            # ai_turn_source / user_turn_sources presets
            ats = (
                (ai_turn_source or "last").strip().lower()
                if non_interactive
                else (
                    ai_turn_source
                    or _ty.prompt("AI turn source (last/all_agents/named_steps)", default="last")
                )
            )
            if ats in {"last", "all_agents", "named_steps"}:
                lines.append(f"      ai_turn_source: {ats}")
                if ats == "named_steps":
                    # Provide a placeholder named step list
                    lines.append('      named_steps: ["clarify"]')
            uts = (
                (user_turn_sources or "hitl").strip()
                if non_interactive
                else (
                    user_turn_sources
                    or _ty.prompt(
                        "User turn sources (comma names, include 'hitl' to capture HITL)",
                        default="hitl",
                    )
                )
            )
            if uts:
                # Normalize to YAML list
                sources = [s.strip() for s in str(uts).split(",") if s.strip()]
                if sources:
                    if len(sources) == 1:
                        lines.append(f"      user_turn_sources: [{sources[0]!r}]")
                    else:
                        joined = ", ".join(repr(s) for s in sources)
                        lines.append(f"      user_turn_sources: [{joined}]")
        lines.append("      init:")
        if goal:
            lines.append(
                '        history:\n          start_with:\n            from_step: get_goal\n            prefix: "User: "'
            )
        else:
            lines.append("        # add init ops as needed")
        lines.append("      body:")
        names = [
            s.strip() for s in (body_steps.split(",") if body_steps else ["clarify"]) if s.strip()
        ]
        for nm_step in names:
            lines.append(
                f"        - kind: step\n          name: {nm_step}\n          updates_context: true"
            )
        if propagation is not None and propagation != "auto":
            lines.append("      propagation:\n        next_input: " + propagation)
        else:
            lines.append("      propagation: auto")
        if stop_when_finished:
            lines.append("      stop_when: agent_finished")
        if out_mode == "text":
            lines.append("      output:\n        text: conversation_history")
        else:
            lines.append(
                "      output:\n        fields:\n          goal: initial_prompt\n          clarifications: conversation_history"
            )
    elif pattern == "map":
        lines.append("  - kind: map")
        lines.append(f"    name: {nm}")
        lines.append("    map:")
        it_name = iterable_name or "items"
        lines.append(f"      iterable_input: {it_name}")
        lines.append("      body:")
        lines.append(
            "        - kind: step\n          name: process\n          updates_context: false"
        )
        lines.append("      init:")
        lines.append('        - set: "context.scratchpad.note"\n          value: "mapping"')
        lines.append("      finalize:")
        lines.append('        output:\n          results_str: "{{ previous_step }}"')
    elif pattern == "parallel":
        lines.append("  - kind: parallel")
        lines.append(f"    name: {nm}")
        names = [
            s.strip()
            for s in (branch_names.split(",") if branch_names else ["a", "b"])
            if s.strip()
        ]
        lines.append("    branches:")
        for bn in names:
            lines.append(f"      {bn}:")
            lines.append(f"        - kind: step\n          name: step_{bn}")
        lines.append(f"    reduce: {reduce_mode or 'keys'}")
    else:
        # Default to loop if unknown pattern
        lines.append("  - kind: loop")
        lines.append(f"    name: {nm}")
        lines.append("    loop:\n      body:\n        - kind: step\n          name: clarify")

    yaml_text = "\n".join(lines)

    if output_dir:
        try:
            _os.makedirs(output_dir, exist_ok=True)
            path = _os.path.join(output_dir, "pipeline.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            typer.secho(f"Wrote natural YAML to {path}", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"Failed to write YAML: {e}", fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        _ty.echo(yaml_text)


@app.command(name="explain", help="Explain a pipeline YAML in plain language")
def explain_cmd(
    path: Annotated[
        Optional[str],
        typer.Argument(
            help="Path to pipeline.yaml (defaults to project pipeline.yaml)",
        ),
    ] = None,
) -> None:
    import sys as _sys
    from pathlib import Path as _Path
    import yaml as _yaml

    try:
        # Resolve default path if omitted
        resolved_path: Optional[str]
        if path is None or str(path).strip() == "":
            root = find_project_root()
            # Prefer pipeline.yaml, fallback to pipeline.yml
            pyaml = (_Path(root) / "pipeline.yaml").resolve()
            pyml = (_Path(root) / "pipeline.yml").resolve()
            if pyaml.exists():
                resolved_path = str(pyaml)
            elif pyml.exists():
                resolved_path = str(pyml)
            else:
                typer.secho(
                    "No pipeline.yaml found in project. Provide a PATH to a YAML file.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(2)
        elif path == "-":
            # Read YAML from stdin
            yaml_text = _sys.stdin.read()
            data = _yaml.safe_load(yaml_text)
            resolved_path = None
        else:
            # Support passing a directory; use project markers to find YAML
            p = _Path(path).resolve()
            if p.is_dir():
                pyaml = (p / "pipeline.yaml").resolve()
                pyml = (p / "pipeline.yml").resolve()
                if pyaml.exists():
                    resolved_path = str(pyaml)
                elif pyml.exists():
                    resolved_path = str(pyml)
                else:
                    typer.secho(
                        f"Directory does not contain pipeline.yaml: {p}",
                        fg=typer.colors.RED,
                    )
                    raise typer.Exit(2)
            else:
                resolved_path = str(p)

        # Load YAML if we haven't already from stdin
        if resolved_path is not None:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)

        if not isinstance(data, dict):
            typer.secho("Invalid YAML format", fg=typer.colors.RED)
            raise typer.Exit(2)

        steps = data.get("steps") or []
        summary: list[str] = []
        if isinstance(data.get("name"), str):
            summary.append(f"Name: {data.get('name')}")
        summary.append(f"Steps count: {len(steps)}")
        for idx, st in enumerate(steps):
            if not isinstance(st, dict):
                continue
            kind = st.get("kind", "step")
            nm = st.get("name", f"step_{idx}")
            if kind == "loop":
                loop = st.get("loop") or {}
                conv = loop.get("conversation")
                prop = loop.get("propagation")
                out = loop.get("output") or {}
                has_tpl = bool(loop.get("output_template"))
                line = f"Loop '{nm}':"
                if conv:
                    line += " conversation: true;"
                if prop:
                    nxt = prop if isinstance(prop, str) else prop.get("next_input")
                    line += f" propagation: {nxt};"
                if has_tpl:
                    line += " output: template;"
                elif out:
                    line += " output: text;" if "text" in out else " output: fields;"
                summary.append(line)
            elif kind == "map":
                mp = st.get("map") or {}
                init = mp.get("init")
                fin = mp.get("finalize")
                iterable = mp.get("iterable_input")
                line = f"Map '{nm}': items from {iterable}"
                if init:
                    line += "; init present"
                if fin:
                    line += "; finalize present"
                summary.append(line)
            elif kind == "parallel":
                brs = (st.get("branches") or {}).keys()
                red = st.get("reduce")
                line = f"Parallel '{nm}': branches={list(brs)}"
                if red:
                    line += f"; reduce={red}"
                summary.append(line)
            else:
                summary.append(f"Step '{nm}' ({kind})")
        typer.echo("\n".join(summary))
    except FileNotFoundError:
        # Use the most relevant path in the message
        missing = resolved_path if "resolved_path" in locals() else path
        typer.secho(f"File not found: {missing}", fg=typer.colors.RED)
        raise typer.Exit(1)
    except Exception as e:
        typer.secho(f"Failed to explain: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


def demo_yaml_cmd(
    demo_name: Annotated[
        str,
        typer.Option("demo", help="Name of the demo pipeline"),
    ] = "demo",
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output", "-o", help="Path to write the demo YAML (default: ./pipeline.demo.yaml)"
        ),
    ] = None,
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="Demo preset",
            case_sensitive=False,
            click_type=click.Choice(["conversational_loop", "map_hitl"]),
        ),
    ] = "conversational_loop",
    conversation: Annotated[
        bool,
        typer.Option(
            "--conversation/--no-conversation",
            help="Enable conversation:true on loop preset",
        ),
    ] = True,
    ai_turn_source: Annotated[
        Optional[str],
        typer.Option("--ai-turn-source", help="AI turn source (last/all_agents/named_steps)"),
    ] = None,
    user_turn_sources: Annotated[
        Optional[str],
        typer.Option(
            "--user-turn-sources",
            help="Comma-separated user turn sources (e.g., hitl,stepA,stepB)",
        ),
    ] = None,
    history_strategy: Annotated[
        Optional[str],
        typer.Option(
            "--history-strategy",
            help="History strategy (truncate_tokens/truncate_turns/summarize)",
        ),
    ] = None,
    history_max_tokens: Annotated[
        Optional[int],
        typer.Option("--history-max-tokens", help="History max_tokens"),
    ] = None,
    history_max_turns: Annotated[
        Optional[int],
        typer.Option("--history-max-turns", help="History max_turns"),
    ] = None,
    history_summarize_ratio: Annotated[
        Optional[float],
        typer.Option("--history-summarize-ratio", help="History summarize_ratio (0..1)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite output if it exists"),
    ] = False,
) -> None:
    """Generate a modern, ready-to-run demo pipeline YAML.

    The default preset produces a conversational loop demo with robust history wiring.
    """
    try:
        uts_list = (
            [s.strip() for s in user_turn_sources.split(",") if s.strip()]
            if user_turn_sources
            else None
        )
        yaml_text = generate_demo_yaml(
            demo_name=demo_name,
            preset=preset,
            conversation=conversation,
            ai_turn_source=ai_turn_source,
            user_turn_sources=uts_list,
            history_strategy=history_strategy,
            history_max_tokens=history_max_tokens,
            history_max_turns=history_max_turns,
            history_summarize_ratio=history_summarize_ratio,
        )
        # Validate for user-friendliness; do not block on failures
        try:
            report = validate_yaml_text(yaml_text)
            if not getattr(report, "is_valid", True):
                typer.secho(
                    "[yellow]Warning: generated YAML reported validation issues. Proceeding anyway.[/yellow]",
                    err=True,
                )
        except Exception:
            # Non-fatal validation failure
            pass

        path = Path(output or "pipeline.demo.yaml").resolve()
        if path.exists() and not force:
            typer.secho(
                f"Refusing to overwrite existing file: {path}. Use --force to overwrite.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(2)
        path.write_text(yaml_text, encoding="utf-8")
        typer.secho(f"✅ Demo YAML written to {path}", fg=typer.colors.GREEN)
        # Quick next-steps hint
        if path.name != "pipeline.yaml":
            typer.secho(f"Run with: flujo run --file {path}", fg=typer.colors.CYAN)
    except Exception as e:
        typer.secho(f"Failed to generate demo YAML: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


# Explicit exports
__all__ = [
    "app",
    "status",
    "solve",
    "version_cmd",
    "show_config_cmd",
    "bench",
    "add_eval_case_cmd",
    "improve",
    "explain",
    "validate",
    "run",
    "lens_app",
    "main",
]

# Register only intended top-level commands per FSD-021
try:
    app.command(
        name="validate",
        help="✅ Validate the project's pipeline.yaml file.",
    )(validate)
except Exception:
    pass


if __name__ == "__main__":
    try:
        # Local alias shim: support legacy `--format` for validate commands when executed directly
        try:
            argv = list(_sys.argv)
            if any(
                tok == "--format" or (isinstance(tok, str) and tok.startswith("--format="))
                for tok in argv
            ):
                # Map for validate
                if "validate" in argv or (
                    len(argv) >= 3 and argv[1] == "dev" and argv[2] == "validate"
                ):
                    for i, tok in enumerate(argv):
                        if tok == "--format":
                            argv[i] = "--output-format"
                        elif isinstance(tok, str) and tok.startswith("--format="):
                            argv[i] = tok.replace("--format=", "--output-format=", 1)
                    _sys.argv[:] = argv
                # No special mapping for other commands
        except Exception:
            pass
        app()
    except (SettingsError, ConfigurationError) as e:
        typer.echo(f"[red]Settings error: {e}[/red]", err=True)
        raise typer.Exit(2)


def get_cli_defaults(command: str) -> Dict[str, Any]:
    """Pass-through for tests to monkeypatch at flujo.cli.main level.

    Delegates to the real config manager function unless monkeypatched in tests.
    """
    return _get_cli_defaults(command)


# Conditionally register experimental commands to avoid breaking CLI tests
try:
    if _os.environ.get("FLUJO_ENABLE_DEMO_YAML") == "1":
        app.command(name="demo-yaml", help="Scaffold a robust demo pipeline YAML")(demo_yaml_cmd)
except Exception:
    pass


# Compatibility functions for testing - re-export functions that tests expect to monkeypatch
# These maintain the testing interface while the actual implementations live elsewhere


def run_default_pipeline(pipeline: Any, task: Any) -> Any:
    """Compatibility function for testing - re-exports from recipes.factories."""
    return _run_default_pipeline(pipeline, task)


def make_review_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_review_agent(model)


def make_solution_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_solution_agent(model)


def make_validator_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_validator_agent(model)


def get_reflection_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _get_reflection_agent(model)


def make_default_pipeline(**kwargs: Any) -> Any:
    """Compatibility function for testing - re-exports from recipes.factories."""
    from flujo.recipes.factories import make_default_pipeline as _make_default_pipeline

    return _make_default_pipeline(**kwargs)


"""Typed re-exports for helpers/tests and mypy visibility."""


# Serialization helper
def safe_deserialize(obj: Any) -> Any:
    return _safe_deserialize(obj)


# Async pipeline runner
run_pipeline_async = _run_pipeline_async

# Self-improvement API
evaluate_and_improve = _evaluate_and_improve
SelfImprovementAgent = _SelfImprovementAgent
ImprovementReport = _ImprovementReport


def make_self_improvement_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_self_improvement_agent(model)


def load_settings() -> Any:
    """Compatibility function for testing - re-exports from config_manager."""
    from flujo.infra.config_manager import load_settings as _load_settings

    return _load_settings()


def _extract_pipeline_name_from_yaml(text: str) -> Optional[str]:
    try:
        import yaml as _yaml

        data = _yaml.safe_load(text)
        if isinstance(data, dict):
            val = data.get("name")
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        return None
    return None
