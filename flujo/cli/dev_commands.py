from __future__ import annotations

# mypy: ignore-errors
from pathlib import Path
from typing import Any, Optional, Union, cast
import json
import os
import typer
import click
from typing_extensions import Annotated
from flujo.exceptions import ConfigurationError
from flujo.infra import telemetry
from .helpers import (
    run_benchmark_pipeline,
    create_benchmark_table,
    setup_solve_command_environment,
    execute_solve_pipeline,
    get_version_string,
    get_masked_settings_dict,
    print_rich_or_typer,
    get_pipeline_explanation,
    ensure_project_root_on_sys_path,
    apply_cli_defaults,
    execute_improve,
    load_mermaid_code,
    load_pipeline_from_file,
    find_project_root,
    validate_pipeline_file,
)
from .config import load_backend_from_config
from flujo.utils.serialization import safe_serialize, safe_deserialize
from flujo.domain.dsl import Pipeline

logfire = telemetry.logfire

# Local alias to avoid circular import with main
ScorerType = str


def register_commands(
    dev_app: typer.Typer, experimental_app: typer.Typer, budgets_app: typer.Typer
) -> None:
    @dev_app.command(name="health-check", help="Analyze AROS signals from recent traces")
    def dev_health_check(
        project: Annotated[
            Optional[str], typer.Option("--project", help="Project root path")
        ] = None,
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
        try:
            from rich.console import Console as _Console

            console = _Console()
        except ModuleNotFoundError:
            console = None
        # Ensure console.print works even without Rich
        if console is None:

            class _PlainConsole:
                def print(self, msg: object, *args: object, **kwargs: object) -> None:
                    from .helpers import print_rich_or_typer as _prt

                    _prt(str(msg))

            console = _PlainConsole()  # type: ignore[assignment]
        if project:
            try:
                ensure_project_root_on_sys_path(Path(project))
            except Exception:
                pass
        try:
            backend = load_backend_from_config()
        except Exception as e:
            from .helpers import print_rich_or_typer

            if console is not None:
                console.print(
                    f"[red]Failed to initialize state backend: {type(e).__name__}: {e}[/red]"
                )
            else:
                print_rich_or_typer(
                    f"[red]Failed to initialize state backend: {type(e).__name__}: {e}[/red]",
                    stderr=True,
                )
            raise typer.Exit(code=1)

        import anyio

        async def _run() -> None:
            runs = await backend.list_runs(pipeline_name=pipeline, limit=limit)
            # Optional time window filter (best-effort)
            from datetime import datetime, timedelta, timezone

            parsed_times: dict[str, datetime] = {}
            cutoff = None
            if since_hours is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=int(since_hours))

                def _parse(ts: object) -> datetime | None:
                    try:
                        # Numeric epoch
                        if isinstance(ts, (int, float)):
                            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        if isinstance(ts, str):
                            s = ts.strip()
                            # Support Z-terminated ISO by normalizing to UTC
                            try:
                                return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
                                    timezone.utc
                                )
                            except Exception:
                                # Try epoch encoded as string
                                return datetime.fromtimestamp(float(s), tz=timezone.utc)
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
                if console is not None:
                    console.print("No runs found.")
                else:
                    from .helpers import print_rich_or_typer

                    print_rich_or_typer("No runs found.")
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
                now = datetime.now(timezone.utc)
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
                            totals["stages"][stage] = int(totals["stages"].get(stage, 0)) + int(
                                v or 0
                            )
                            run_stage_counts[stage] = int(run_stage_counts.get(stage, 0)) + int(
                                v or 0
                            )
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
                                transforms_count[str(tname)] = (
                                    transforms_count.get(str(tname), 0) + 1
                                )
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
                                        max(
                                            0, int(offset_seconds / (total_seconds / len(buckets)))
                                        ),
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
            console.print(
                f"Total coercions: {totals['coercion_total']} (stages: {totals['stages']})"
            )
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
                        top_stages = sorted(agg_stage.items(), key=lambda kv: kv[1], reverse=True)[
                            :3
                        ]
                        stage_names = ", ".join(f"{k}:{v}" for k, v in top_stages)
                        console.print(f"Top coercion stages across buckets: {stage_names}")
            # Top 10 steps by coercions
            top = sorted(per_step.items(), key=lambda kv: kv[1].get("coercions", 0), reverse=True)[
                :10
            ]
            if top:
                console.print("\n[bold]Top steps by coercions[/bold]")
                for name, stats in top:
                    console.print(f"- {name}: {stats['coercions']}")
            # Top 10 models by coercions
            topm = sorted(
                per_model.items(), key=lambda kv: kv[1].get("coercions", 0), reverse=True
            )[:10]
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
                            for k, v in sorted(
                                stages_map.items(), key=lambda kv: kv[1], reverse=True
                            )[:3]
                        )
                        console.print(f"- {name}: {parts}")
            if topm:
                console.print("\n[bold]Stage breakdowns by model[/bold]")
                for name, _ in topm[:3]:
                    stages_map = per_model_stages.get(name, {})
                    if stages_map:
                        parts = ", ".join(
                            f"{k}:{v}"
                            for k, v in sorted(
                                stages_map.items(), key=lambda kv: kv[1], reverse=True
                            )[:3]
                        )
                        console.print(f"- {name}: {parts}")

            # Simple recommendations
            if console is not None:
                console.print("\n[bold]Recommendations[/bold]")
            else:
                from .helpers import print_rich_or_typer

                print_rich_or_typer("\n[bold]Recommendations[/bold]")
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
                                    max(0, series[i + 1] - series[i])
                                    for i in range(len(series) - 1)
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
                                    max(0, series[i + 1] - series[i])
                                    for i in range(len(series) - 1)
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
                            dt_ext = int(lmap.get("extract", 0) or 0) - int(
                                pmap.get("extract", 0) or 0
                            )
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
                            dt_ext = int(lmap.get("extract", 0) or 0) - int(
                                pmap.get("extract", 0) or 0
                            )
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
                            "generated_at": datetime.now(timezone.utc).isoformat(),
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
            import importlib

            cli_main = importlib.import_module("flujo.cli.main")
            setup_fn = getattr(
                cli_main, "setup_solve_command_environment", setup_solve_command_environment
            )
            exec_fn = getattr(cli_main, "execute_solve_pipeline", execute_solve_pipeline)
            load_settings_fn = getattr(cli_main, "load_settings", None)
            if load_settings_fn is None:
                from flujo.infra.config_manager import load_settings as load_settings_fn  # type: ignore

            # Set up command environment using helper function
            cli_args, metadata, agents = setup_fn(
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
            settings = load_settings_fn()

            # Execute pipeline using helper function
            best = exec_fn(
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
            import importlib

            cli_main = importlib.import_module("flujo.cli.main")
            bench_fn = getattr(cli_main, "run_benchmark_pipeline", run_benchmark_pipeline)
            table_fn = getattr(cli_main, "create_benchmark_table", create_benchmark_table)

            # Apply CLI defaults from configuration file
            cli_args = apply_cli_defaults("bench", rounds=rounds)
            rounds = cast(int, cli_args["rounds"])

            # Run benchmark using helper function
            times, scores = bench_fn(prompt, rounds, logfire)

            # Create and display results table using helper function
            table = table_fn(times, scores)
            try:
                from rich.console import Console as _Console

                _Console().print(table)
            except ModuleNotFoundError:
                from .helpers import print_rich_or_typer

                print_rich_or_typer(str(table))
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
            import importlib

            cli_main = importlib.import_module("flujo.cli.main")
            improve_fn = getattr(cli_main, "execute_improve", execute_improve)
            output = improve_fn(
                pipeline_path=pipeline_path,
                dataset_path=dataset_path,
                improvement_agent_model=improvement_agent_model,
                json_output=json_output,
            )
            if json_output and output is not None:
                typer.echo(output)

        except Exception as e:
            print_rich_or_typer(f"[red]Error running improvement: {e}", stderr=True)
            raise typer.Exit(1) from e

    @dev_app.command(name="explain")
    def explain(path: str) -> None:
        """
        Print a summary of a pipeline defined in a file.

        Args:
            path: Path to the pipeline definition file

        Returns:
            None: Prints pipeline step explanations to stdout

        Raises:
            typer.Exit: If there is an error loading the pipeline file
        """
        try:
            for explanation in get_pipeline_explanation(path):
                typer.echo(explanation)
        except Exception as e:
            print_rich_or_typer(f"[red]Failed to load pipeline file: {e}", stderr=True)
            raise typer.Exit(1) from e

    def _validate_impl(
        path: Optional[str],
        strict: bool,
        output_format: str,
        *,
        include_imports: bool = True,
        fail_on_warn: bool = False,
        rules: Optional[str] = None,
        explain: bool = False,
        baseline: Optional[str] = None,
        update_baseline: bool = False,
        fix: bool = False,
        yes: bool = False,
        fix_rules: Optional[str] = None,
        fix_dry_run: bool = False,
    ) -> None:
        from .exit_codes import EX_VALIDATION_FAILED, EX_IMPORT_ERROR, EX_RUNTIME_ERROR
        import traceback as _tb
        import os as _os

        try:
            if path is None:
                root = find_project_root()
                path = str((Path(root) / "pipeline.yaml").resolve())
            # Preload linter rule overrides so early-skips match CLI --rules
            _preloaded_mapping: dict[str, str] | None = None

            def _load_rules_mapping_from_file(rules_path: str) -> dict[str, str] | None:
                import os as _os
                import json as _json

                try:
                    if not _os.path.exists(rules_path):
                        return None
                    try:
                        with open(rules_path, "r", encoding="utf-8") as f:
                            data = _json.load(f)
                        return (
                            {str(k).upper(): str(v).lower() for k, v in data.items()}
                            if isinstance(data, dict)
                            else None
                        )
                    except Exception:
                        try:
                            import tomllib as _tomllib
                        except Exception:  # pragma: no cover
                            import tomli as _tomllib  # type: ignore
                        with open(rules_path, "rb") as f:
                            data = _tomllib.load(f)
                        if isinstance(data, dict):
                            vm = (
                                data.get("validation", {}).get("rules")
                                if isinstance(data.get("validation"), dict)
                                else data
                            )
                            if isinstance(vm, dict):
                                return {str(k).upper(): str(v).lower() for k, v in vm.items()}
                except Exception:
                    return None
                return None

            # If rules provided, preload into linters
            if rules:
                # Load rule overrides for linters through env; avoid direct imports
                mapping = None
                if not os.path.exists(rules):
                    # Profile name will be handled by linters via FLUJO_RULES_PROFILE
                    os.environ["FLUJO_RULES_PROFILE"] = rules
                else:
                    mapping = _load_rules_mapping_from_file(rules)
                    if mapping:
                        # Prefer setting env JSON for child processes/linters
                        os.environ["FLUJO_RULES_JSON"] = json.dumps(mapping)
                _preloaded_mapping = mapping

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

            # Optional: apply safe auto-fixes before printing results
            applied_fixes_metrics: dict[str, Any] | None = None
            if fix:
                try:
                    from ..validation.fixers import plan_fixes, apply_fixes_to_file, build_fix_patch

                    # Parse per-rule filter from env (comma-separated globs), e.g., "V-T1,V-C2*"
                    rule_filter = None
                    try:
                        if fix_rules:
                            rule_filter = [x.strip() for x in fix_rules.split(",") if x.strip()]
                        else:
                            rf = _os.getenv("FLUJO_FIX_RULES")
                            if rf:
                                rule_filter = [x.strip() for x in rf.split(",") if x.strip()]
                    except Exception:
                        rule_filter = None

                    plan = plan_fixes(path, report, rules=rule_filter)
                    if plan:
                        # Preview
                        try:
                            from rich.console import Console

                            con = Console(stderr=True, highlight=False)
                            con.print("[cyan]Auto-fix preview:[/cyan]")
                            for item in plan:
                                con.print(
                                    f"  - {item['rule_id']}: {item['count']} change(s) — {item['title']}"
                                )
                            if fix_dry_run:
                                patch, metrics = build_fix_patch(path, report, rules=rule_filter)
                                if patch:
                                    con.print("[cyan]Patch (dry-run):[/cyan]")
                                    con.print(patch)
                                applied_fixes_metrics = metrics
                                # Do not apply when dry-run
                                do_apply = False
                        except Exception:
                            pass
                        # Confirm
                        if not fix_dry_run:
                            do_apply = yes
                            try:
                                # In JSON output mode, avoid interactive prompt unless --yes
                                if output_format == "json" and not yes:
                                    do_apply = False
                                elif not yes:
                                    from typer import confirm as _confirm

                                    do_apply = _confirm("Apply these changes?", default=True)
                            except Exception:
                                do_apply = yes

                        if do_apply:
                            applied, backup, metrics = apply_fixes_to_file(
                                path, report, assume_yes=yes, rules=rule_filter
                            )
                            # Re-validate to show updated report
                            if applied:
                                try:
                                    report = validate_pipeline_file(
                                        path, include_imports=include_imports
                                    )
                                    # Re-apply rules/profile mapping to new report
                                    if profile_mapping:
                                        report = _apply_mapping(report, profile_mapping)
                                    else:
                                        report = _apply_rules(report, rules)
                                except Exception:
                                    pass
                            # Emit brief metrics to stderr
                            try:
                                from rich.console import Console as _C

                                _C(stderr=True).print(
                                    f"[green]Applied {metrics.get('total_applied', 0)} change(s). Backup: {backup or 'n/a'}[/green]"
                                )
                            except Exception:
                                pass
                            # Preserve metrics for JSON output
                            applied_fixes_metrics = metrics
                        else:
                            applied_fixes_metrics = {"applied": {}, "total_applied": 0}
                    else:
                        try:
                            from rich.console import Console

                            Console(stderr=True).print("[yellow]No fixable issues found.[/yellow]")
                        except Exception:
                            pass
                        applied_fixes_metrics = {"applied": {}, "total_applied": 0}
                except Exception as e:
                    # Fixers are best-effort; capture and continue
                    try:
                        from ..infra.telemetry import logfire as _lf

                        _lf.debug(
                            f"[validate] Auto-fix flow suppressed due to: {type(e).__name__}: {e}"
                        )
                    except Exception:
                        pass
                    applied_fixes_metrics = {"applied": {}, "total_applied": 0}
            else:
                applied_fixes_metrics = None

            # Optional baseline delta handling (compare post-rules report to previous)
            baseline_info: dict[str, Any] | None = None
            if baseline:
                try:
                    import os as _os

                    if _os.path.exists(baseline):
                        with open(baseline, "r", encoding="utf-8") as bf:
                            prev_raw = json.load(bf)
                    else:
                        prev_raw = None

                    def _key_of(d: dict[str, Any]) -> tuple[str, str]:
                        return (str(d.get("rule_id", "")).upper(), str(d.get("step_name", "")))

                    cur_err = [e.model_dump() for e in report.errors]
                    cur_warn = [w.model_dump() for w in report.warnings]
                    if isinstance(prev_raw, dict):
                        prev_err = [
                            x for x in (prev_raw.get("errors") or []) if isinstance(x, dict)
                        ]
                        prev_warn = [
                            x for x in (prev_raw.get("warnings") or []) if isinstance(x, dict)
                        ]
                    else:
                        prev_err, prev_warn = [], []

                    prev_err_keys = {_key_of(x) for x in prev_err}
                    prev_warn_keys = {_key_of(x) for x in prev_warn}
                    cur_err_keys = {_key_of(x) for x in cur_err}
                    cur_warn_keys = {_key_of(x) for x in cur_warn}

                    added_errors = [x for x in cur_err if _key_of(x) not in prev_err_keys]
                    added_warnings = [x for x in cur_warn if _key_of(x) not in prev_warn_keys]
                    removed_errors = [x for x in prev_err if _key_of(x) not in cur_err_keys]
                    removed_warnings = [x for x in prev_warn if _key_of(x) not in cur_warn_keys]

                    # Replace the visible report (and therefore exit-code semantics) with post-baseline view
                    from flujo.domain.pipeline_validation import (
                        ValidationReport as _VR,
                        ValidationFinding as _VF,
                    )

                    def _vf_list(arr: list[dict[str, Any]]) -> list[_VF]:
                        out: list[_VF] = []
                        for it in arr:
                            try:
                                out.append(_VF(**it))
                            except Exception:
                                continue
                        return out

                    report = _VR(errors=_vf_list(added_errors), warnings=_vf_list(added_warnings))
                    baseline_info = {
                        "applied": True,
                        "file": baseline,
                        "added": {"errors": added_errors, "warnings": added_warnings},
                        "removed": {"errors": removed_errors, "warnings": removed_warnings},
                    }
                except Exception:
                    baseline_info = {"applied": False, "file": baseline}

            # Optional explanation catalog for rules (centralized)
            def _explain(rule_id: str) -> str | None:
                try:
                    from ..validation.rules_catalog import get_rule

                    info = get_rule(rule_id)
                    return info.description if info else None
                except Exception:
                    return None

            # Optional telemetry: counts per severity/rule when enabled
            telemetry_counts: dict[str, dict[str, int]] | None = None
            try:
                if os.getenv("FLUJO_CLI_TELEMETRY") == "1":
                    from collections import Counter

                    err = Counter([e.rule_id for e in report.errors])
                    warn = Counter([w.rule_id for w in report.warnings])
                    telemetry_counts = {
                        "error": dict(err),
                        "warning": dict(warn),
                    }
            except Exception:
                telemetry_counts = None

            # Duplicate fixer block removed; the unified fixer flow above handles preview,
            # dry-run, apply, and metrics consistently.

            if output_format == "json":
                # Emit machine-friendly JSON (errors, warnings, is_valid)
                payload = {
                    "is_valid": bool(report.is_valid),
                    "errors": [
                        (
                            {
                                **e.model_dump(),
                                **({"explain": _explain(e.rule_id)} if explain else {}),
                            }
                        )
                        for e in report.errors
                    ],
                    "warnings": [
                        (
                            {
                                **w.model_dump(),
                                **({"explain": _explain(w.rule_id)} if explain else {}),
                            }
                        )
                        for w in report.warnings
                    ],
                    "path": path,
                    **({"baseline": baseline_info} if baseline_info else {}),
                    **({"counts": telemetry_counts} if telemetry_counts else {}),
                    **(
                        {"fixes": applied_fixes_metrics}
                        if applied_fixes_metrics is not None
                        else {}
                    ),
                    **({"fixes_dry_run": True} if fix and fix_dry_run else {}),
                }
                typer.echo(json.dumps(payload))
            elif output_format == "sarif":
                # Minimal SARIF 2.1.0 conversion
                def _level(sev: str) -> str:
                    return "error" if sev == "error" else "warning"

                rules_index: dict[str, int] = {}
                sarif_rules: list[dict[str, Any]] = []
                sarif_results: list[dict[str, Any]] = []

                def _append_rule(info: Any, rid: str | None = None) -> None:
                    rule_id = (rid or getattr(info, "id", "") or "").upper()
                    if not rule_id or rule_id in rules_index:
                        return
                    sarif_rules.append(
                        {
                            "id": rule_id,
                            "name": (getattr(info, "title", None) or rule_id),
                            "shortDescription": {"text": (getattr(info, "title", None) or rule_id)},
                            **(
                                {"fullDescription": {"text": getattr(info, "description")}}
                                if (hasattr(info, "description") and getattr(info, "description"))
                                else {}
                            ),
                            **(
                                {"helpUri": getattr(info, "help_uri")}
                                if (hasattr(info, "help_uri") and getattr(info, "help_uri"))
                                else {
                                    "helpUri": f"https://aandresalvarez.github.io/flujo/reference/validation_rules/#{rule_id.lower()}"
                                }
                            ),
                        }
                    )
                    rules_index[rule_id] = len(sarif_rules) - 1

                def _rule_ref(rule_id: str) -> dict[str, Any]:
                    rid = (rule_id or "").upper()
                    if rid not in rules_index:
                        try:
                            from ..validation.rules_catalog import get_rule

                            info = get_rule(rid)
                        except Exception:
                            info = None
                        _append_rule(info, rid)
                    return {"ruleId": rid}

                # Preload the entire catalog so metadata is present even when there are zero findings.
                try:
                    from ..validation import rules_catalog as _rules_catalog

                    for _rule in getattr(_rules_catalog, "_CATALOG", {}).values():
                        _append_rule(_rule)
                except Exception:
                    # Best-effort; fall back to lazy rule additions when findings reference them.
                    pass

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
                    print_rich_or_typer("[red]Validation errors detected[/red]:")
                    print_rich_or_typer(
                        "See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/",
                        style="red",
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
                    print_rich_or_typer("[yellow]Warnings[/yellow]:")
                    print_rich_or_typer(
                        "See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/",
                        style="yellow",
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
                    print_rich_or_typer("[green]Pipeline is valid[/green]")
                if telemetry_counts:
                    try:
                        total_e = sum(telemetry_counts.get("error", {}).values())
                        total_w = sum(telemetry_counts.get("warning", {}).values())
                        print_rich_or_typer(
                            f"[cyan]Counts[/cyan]: errors={total_e}, warnings={total_w}"
                        )
                    except Exception:
                        pass
                if baseline_info and baseline_info.get("applied"):
                    try:
                        ae = len(baseline_info["added"]["errors"])  # type: ignore[index]
                        aw = len(baseline_info["added"]["warnings"])  # type: ignore[index]
                        re_ = len(baseline_info["removed"]["errors"])  # type: ignore[index]
                        rw = len(baseline_info["removed"]["warnings"])  # type: ignore[index]
                        msg = f"Baseline applied: +{ae} errors, +{aw} warnings; removed: -{re_} errors, -{rw} warnings"
                        print_rich_or_typer(f"[cyan]{msg}[/cyan]")
                    except Exception:
                        pass

            # Optionally write/update the baseline file with the current (post-baseline) view
            if baseline and update_baseline:
                try:
                    with open(baseline, "w", encoding="utf-8") as bf:
                        json.dump(
                            {
                                "errors": [e.model_dump() for e in report.errors],
                                "warnings": [w.model_dump() for w in report.warnings],
                            },
                            bf,
                        )
                except Exception:
                    pass

            if strict and not report.is_valid:
                raise typer.Exit(EX_VALIDATION_FAILED)
            if fail_on_warn and report.warnings:
                raise typer.Exit(EX_VALIDATION_FAILED)
        except ModuleNotFoundError as e:
            # Improve import error messaging with hint on project root
            mod = getattr(e, "name", None) or str(e)
            print_rich_or_typer(
                f"[red]Import error: module '{mod}' not found. Try PYTHONPATH=. or use --project/FLUJO_PROJECT_ROOT[/red]",
                stderr=True,
            )
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                typer.echo("\nTraceback:", err=True)
                typer.echo("".join(_tb.format_exception(e)), err=True)
            raise typer.Exit(EX_IMPORT_ERROR) from e
        except typer.Exit:
            # Preserve intended exit status (e.g., EX_VALIDATION_FAILED)
            raise
        except Exception as e:
            print_rich_or_typer(
                f"[red]Validation failed: {type(e).__name__}: {e}[/red]", stderr=True
            )
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                typer.echo("\nTraceback:", err=True)
                typer.echo("".join(_tb.format_exception(e)), err=True)
            raise typer.Exit(EX_RUNTIME_ERROR) from e

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
                "--rules",
                help="Path to rules JSON/TOML that overrides severities (off/warning/error)",
            ),
        ] = None,
        explain: Annotated[
            bool,
            typer.Option("--explain", help="Include brief 'why this matters' guidance in output"),
        ] = False,
        baseline: Annotated[
            Optional[str],
            typer.Option(
                "--baseline", help="Path to a previous JSON report to compute deltas against"
            ),
        ] = None,
        update_baseline: Annotated[
            bool,
            typer.Option(
                "--update-baseline",
                help="Write the current report (post-baseline view) to --baseline path",
            ),
        ] = False,
        fix: Annotated[
            bool, typer.Option("--fix", help="Apply safe, opt-in auto-fixes (currently V-T1)")
        ] = False,
        yes: Annotated[
            bool, typer.Option("--yes", help="Assume yes to prompts when using --fix")
        ] = False,
        fix_rules: Annotated[
            Optional[str],
            typer.Option(
                "--fix-rules",
                help="Comma-separated list of fixer rules/globs (e.g., V-T1,V-C2*)",
            ),
        ] = None,
        fix_dry_run: Annotated[
            bool,
            typer.Option("--fix-dry-run", help="Preview patch without writing changes"),
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
            baseline=baseline,
            update_baseline=update_baseline,
            fix=fix,
            yes=yes,
            fix_rules=fix_rules,
            fix_dry_run=fix_dry_run,
        )

    @dev_app.command(name="compile-yaml")
    def compile(  # type: ignore[override]
        src: str = typer.Argument(..., help="Input spec: .yaml/.yml or .py"),
        out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file path (.yaml)"),
        normalize: bool = typer.Option(
            True, "--normalize/--no-normalize", help="Normalize YAML formatting and structure"
        ),
    ) -> None:
        """Compile a pipeline spec between YAML and DSL."""
        try:
            if src.endswith((".yaml", ".yml")):
                pipe = Pipeline.from_yaml_file(src)
                yaml_text = pipe.to_yaml() if normalize else Path(src).read_text()
            else:
                pipeline_obj, _ = load_pipeline_from_file(src)
                yaml_text = pipeline_obj.to_yaml()
            if out:
                Path(out).write_text(yaml_text, encoding="utf-8")
                print_rich_or_typer(f"[green]Wrote: {out}")
            else:
                typer.echo(yaml_text)
        except Exception as e:  # noqa: BLE001
            print_rich_or_typer(f"[red]Failed to compile: {e}", stderr=True)
            raise typer.Exit(1) from e

    @budgets_app.command("show")
    def budgets_show(pipeline_name: str) -> None:
        """Print the effective budget for a pipeline and its resolution source."""
        try:
            from flujo.infra.config_manager import ConfigManager
            from flujo.infra.budget_resolver import resolve_limits_for_pipeline

            cfg = ConfigManager().load_config()
            limits, src = resolve_limits_for_pipeline(getattr(cfg, "budgets", None), pipeline_name)

            if limits is None:
                typer.echo("No budget configured (unlimited). Source: none")
                return

            cost = (
                f"${limits.total_cost_usd_limit:.2f}"
                if limits.total_cost_usd_limit is not None
                else "unlimited"
            )
            tokens = (
                f"{limits.total_tokens_limit}"
                if limits.total_tokens_limit is not None
                else "unlimited"
            )
            origin = src.source if src.pattern is None else f"{src.source}[{src.pattern}]"
            typer.echo(f"Effective budget for '{pipeline_name}':")
            typer.echo(f"  - total_cost_usd_limit: {cost}")
            typer.echo(f"  - total_tokens_limit: {tokens}")
            typer.echo(f"Resolved from {origin} in flujo.toml")
        except Exception as e:  # noqa: BLE001
            print_rich_or_typer(f"[red]Failed to resolve budgets: {e}", stderr=True)
            raise typer.Exit(1) from e

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
        """Output a pipeline's Mermaid diagram at the chosen detail level."""
        try:
            mermaid_code = load_mermaid_code(file, object_name, detail_level)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write("```mermaid\n")
                    f.write(mermaid_code)
                    f.write("\n```")
                print_rich_or_typer(f"[green]Mermaid diagram written to {output}")
            else:
                typer.echo("```mermaid")
                typer.echo(mermaid_code)
                typer.echo("```")
        except Exception as e:  # noqa: BLE001
            print_rich_or_typer(f"[red]Failed to load file: {e}", stderr=True)
            raise typer.Exit(1) from e

    # Expose command functions for tests that import from flujo.cli.main
    globals().update(
        {
            "dev_health_check": dev_health_check,
            "solve": solve,
            "version_cmd": version_cmd,
            "show_config_cmd": show_config_cmd,
            "bench": bench,
            "add_eval_case_cmd": add_eval_case_cmd,
            "improve": improve,
            "explain": explain,
            "compile": compile,
            "pipeline_mermaid_cmd": pipeline_mermaid_cmd,
            "budgets_show": budgets_show,
            "validate_dev": validate_dev,
        }
    )


__all__ = ["register_commands"]
