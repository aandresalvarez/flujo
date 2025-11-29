from __future__ import annotations
# mypy: ignore-errors

from pathlib import Path
from typing import Any, Optional

import click
import os
import typer
from typing_extensions import Annotated

from flujo.infra import telemetry
from .config import load_backend_from_config as _load_backend_from_config
from .helpers import ensure_project_root_on_sys_path, get_version_string

logfire = telemetry.logfire


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
        try:
            from flujo.cli import dev_commands as _dev

            backend_factory = getattr(_dev, "load_backend_from_config", _load_backend_from_config)
        except Exception:
            backend_factory = _load_backend_from_config
        backend = backend_factory()
    except Exception as e:
        from .helpers import print_rich_or_typer

        if console is not None:
            console.print(f"[red]Failed to initialize state backend: {type(e).__name__}: {e}[/red]")
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


def register_health_commands(dev_app: typer.Typer) -> None:
    dev_app.command(name="health-check", help="Analyze AROS signals from recent traces")(
        dev_health_check
    )
