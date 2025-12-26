from pathlib import Path
from typing import Optional, TypedDict

import json
import os
import typer
from typing_extensions import Annotated
import traceback as _tb

from .helpers import find_project_root, print_rich_or_typer
from .exit_codes import EX_VALIDATION_FAILED, EX_IMPORT_ERROR, EX_RUNTIME_ERROR

from flujo.domain.services.validation_service import ValidationService
from flujo.domain.pipeline_validation import ValidationFinding
from flujo.infra.reporting.sarif import SarifGenerator


class _BaselineIssueSection(TypedDict):
    errors: list[dict[str, object]]
    warnings: list[dict[str, object]]


class _BaselineInfo(TypedDict):
    applied: bool
    file: str | None
    added: _BaselineIssueSection
    removed: _BaselineIssueSection


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
    try:
        service = ValidationService()

        if path is None:
            root = find_project_root()
            path = str((Path(root) / "pipeline.yaml").resolve())

        # Preload rules mapping (handling environment variables for linters)
        mapping = None
        if rules:
            if not os.path.exists(rules):
                # Profile name
                os.environ["FLUJO_RULES_PROFILE"] = rules
                # Load profile mapping if available
                from ..infra.config_manager import get_config_manager as _cfg

                try:
                    cfg = _cfg().load_config()
                    val = getattr(cfg, "validation", None)
                    profiles = getattr(val, "profiles", None)
                    if profiles and rules in profiles:
                        raw = profiles[rules]
                        if isinstance(raw, dict):
                            mapping = {str(k): str(v) for k, v in raw.items()}
                except Exception:
                    pass
            else:
                # File path
                mapping = service.load_rules_mapping(rules)
                if mapping:
                    os.environ["FLUJO_RULES_JSON"] = json.dumps(mapping)

        # 1. Run Validation
        report = service.validate_file(
            path, include_imports=include_imports, profile_mapping=mapping
        )

        # 2. Compute Baseline Delta (if applicable)
        baseline_info: _BaselineInfo | None = None
        if baseline:
            report, delta = service.compute_baseline_delta(report, baseline)
            baseline_info = {
                "applied": delta.applied,
                "file": delta.file,
                "added": {
                    "errors": [dict(e.model_dump()) for e in delta.added_errors],
                    "warnings": [dict(w.model_dump()) for w in delta.added_warnings],
                },
                "removed": {
                    "errors": [dict(e.model_dump()) for e in delta.removed_errors],
                    "warnings": [dict(w.model_dump()) for w in delta.removed_warnings],
                },
            }

        if update_baseline and baseline:
            with open(baseline, "w", encoding="utf-8") as bf:
                json.dump(
                    {
                        "errors": [e.model_dump() for e in report.errors],
                        "warnings": [w.model_dump() for w in report.warnings],
                    },
                    bf,
                )

        # 3. Auto-fix (Interactive CLI logic)
        applied_metrics = None
        if fix or fix_dry_run:
            # This logic remains in CLI as it is highly interactive and requires UI
            # We import fixers directly.
            from ..validation.fixers import plan_fixes, apply_fixes_to_file, build_fix_patch
            from rich.console import Console

            rules_filter = None
            if fix_rules:
                rules_filter = [x.strip() for x in fix_rules.split(",") if x.strip()]

            plan = plan_fixes(path, report, rules=rules_filter)
            if plan:
                con = Console(stderr=True, highlight=False)
                con.print("[cyan]Auto-fix preview:[/cyan]")
                for item in plan:
                    con.print(f"  - {item['rule_id']}: {item['count']} change(s) — {item['title']}")

                do_apply = False
                if fix_dry_run:
                    patch, metrics = build_fix_patch(path, report, rules=rules_filter)
                    if patch:
                        con.print("[cyan]Patch (dry-run):[/cyan]")
                        con.print(patch)
                    applied_metrics = metrics
                    # Ensure dry run metric flag is set for tests
                    if applied_metrics is None:
                        applied_metrics = {}
                    applied_metrics["fixes_dry_run"] = True
                else:
                    do_apply = yes
                    if not do_apply and output_format != "json":
                        try:
                            from typer import confirm

                            do_apply = confirm("Apply these changes?", default=True)
                        except Exception:
                            pass

                if do_apply:
                    applied, backup, metrics = apply_fixes_to_file(
                        path, report, assume_yes=True, rules=rules_filter
                    )
                    applied_metrics = metrics
                    if applied:
                        # Re-validate!
                        full_report = service.validate_file(
                            path, include_imports=include_imports, profile_mapping=mapping
                        )
                        report = full_report
                        # Note: We lose baseline filtering here if we re-run validaton.
                        # This is acceptable (fixes might have fixed the baseline issues too).
                        if output_format != "json":
                            print_rich_or_typer(
                                f"[green]Applied {metrics.get('total_applied', 0)} changes.[/green]"
                            )
                else:
                    applied_metrics = {"applied": {}, "total_applied": 0}
            else:
                applied_metrics = {"applied": {}, "total_applied": 0, "fixes_dry_run": fix_dry_run}
                if output_format != "json":
                    print_rich_or_typer("[yellow]No fixable issues found.[/yellow]", stderr=True)

        # 4. Output Generation
        if output_format == "json":
            explanation: list[str] | None = None
            if explain and path:
                from .helpers import get_pipeline_explanation

                try:
                    explanation = get_pipeline_explanation(path)
                except Exception:
                    explanation = []

            # Helper to enrich with explanation if needed
            rule_cache: dict[str, str] = {}

            def _dump(f: ValidationFinding, is_error: bool = False) -> dict[str, object]:
                _ = is_error
                d: dict[str, object] = dict(f.model_dump())
                if explain:
                    rid = str(f.rule_id).upper()
                    if rid not in rule_cache:
                        desc = ""
                        try:
                            from flujo.validation.rules_catalog import get_rule

                            info = get_rule(rid)
                            if info:
                                desc = getattr(info, "description", "") or getattr(
                                    info, "title", ""
                                )
                        except ImportError:
                            pass
                        rule_cache[rid] = desc
                    d["explain"] = rule_cache[rid]
                return d

            def _counts(
                errs: list[ValidationFinding], warns: list[ValidationFinding]
            ) -> dict[str, dict[str, int]]:
                c: dict[str, dict[str, int]] = {"error": {}, "warning": {}}
                for e in errs:
                    c["error"][e.rule_id] = c["error"].get(e.rule_id, 0) + 1
                for w in warns:
                    c["warning"][w.rule_id] = c["warning"].get(w.rule_id, 0) + 1
                return c

            payload = {
                "is_valid": bool(report.is_valid),
                "errors": [_dump(e, True) for e in report.errors],
                "warnings": [_dump(w, False) for w in report.warnings],
                "path": path,
                "baseline": baseline_info,
                "fixes": applied_metrics,
                "explanation": explanation,
                "counts": _counts(report.errors, report.warnings),
                "fixes_dry_run": fix_dry_run,
            }
            typer.echo(json.dumps(payload, indent=2))

        elif output_format == "sarif":
            sarif = SarifGenerator().to_sarif(report)
            typer.echo(sarif)

        else:
            # Text Output
            if report.errors:
                print_rich_or_typer("[red]Validation errors detected[/red]:")
                for e in report.errors:
                    loc = f"{e.step_name}: " if e.step_name else ""
                    print_rich_or_typer(f"- [{e.rule_id}] {loc}{e.message}")
                    if e.suggestion:
                        # Indent the suggestion for readability
                        suggestion_lines = e.suggestion.split("\n")
                        for line in suggestion_lines:
                            print_rich_or_typer(f"  {line}")
            if report.warnings:
                print_rich_or_typer("[yellow]Warnings[/yellow]:")
                for w in report.warnings:
                    loc = f"{w.step_name}: " if w.step_name else ""
                    print_rich_or_typer(f"- [{w.rule_id}] {loc}{w.message}")
                    if w.suggestion:
                        # Indent the suggestion for readability
                        suggestion_lines = w.suggestion.split("\n")
                        for line in suggestion_lines:
                            print_rich_or_typer(f"  {line}")
            if report.is_valid:
                print_rich_or_typer("[green]Pipeline is valid[/green]")

            if baseline_info and baseline_info.get("applied"):
                ae = len(baseline_info["added"]["errors"])
                aw = len(baseline_info["added"]["warnings"])
                re_ = len(baseline_info["removed"]["errors"])
                rw = len(baseline_info["removed"]["warnings"])
                print_rich_or_typer(
                    f"[cyan]Baseline applied: +{ae} errors, +{aw} warnings; removed: -{re_} errors, -{rw} warnings[/cyan]"
                )

        # 5. Exit Code
        if strict and not report.is_valid:
            raise typer.Exit(EX_VALIDATION_FAILED)
        if fail_on_warn and report.warnings:
            raise typer.Exit(EX_VALIDATION_FAILED)

    except ModuleNotFoundError as e:
        mod = getattr(e, "name", None) or str(e)
        print_rich_or_typer(f"[red]Import error: module '{mod}' not found.[/red]", stderr=True)
        if os.getenv("FLUJO_CLI_VERBOSE"):
            typer.echo("".join(_tb.format_exception(e)), err=True)
        raise typer.Exit(EX_IMPORT_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_rich_or_typer(f"[red]Validation failed: {type(e).__name__}: {e}[/red]", stderr=True)
        if os.getenv("FLUJO_CLI_VERBOSE"):
            typer.echo("".join(_tb.format_exception(e)), err=True)
        raise typer.Exit(EX_RUNTIME_ERROR) from e


def validate_dev(
    path: Optional[str] = typer.Argument(None, help="Path to pipeline file"),
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
    output_format: Annotated[str, typer.Option("--format", "--output-format")] = "text",
    imports: Annotated[bool, typer.Option("--imports/--no-imports")] = True,
    fail_on_warn: Annotated[bool, typer.Option("--fail-on-warn")] = False,
    rules: Annotated[Optional[str], typer.Option("--rules")] = None,
    explain: Annotated[bool, typer.Option("--explain")] = False,
    baseline: Annotated[Optional[str], typer.Option("--baseline")] = None,
    update_baseline: Annotated[bool, typer.Option("--update-baseline")] = False,
    fix: Annotated[bool, typer.Option("--fix")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    fix_rules: Annotated[Optional[str], typer.Option("--fix-rules")] = None,
    fix_dry_run: Annotated[bool, typer.Option("--fix-dry-run")] = False,
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


def validate(
    path: Optional[str] = typer.Argument(None, help="Path to pipeline file"),
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
    output_format: Annotated[str, typer.Option("--format", "--output-format")] = "text",
    imports: Annotated[bool, typer.Option("--imports/--no-imports")] = True,
    fail_on_warn: Annotated[bool, typer.Option("--fail-on-warn")] = False,
    rules: Annotated[Optional[str], typer.Option("--rules")] = None,
    explain: Annotated[bool, typer.Option("--explain")] = False,
    baseline: Annotated[Optional[str], typer.Option("--baseline")] = None,
    update_baseline: Annotated[bool, typer.Option("--update-baseline")] = False,
    fix: Annotated[bool, typer.Option("--fix")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    fix_rules: Annotated[Optional[str], typer.Option("--fix-rules")] = None,
    fix_dry_run: Annotated[bool, typer.Option("--fix-dry-run")] = False,
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
        baseline=baseline,
        update_baseline=update_baseline,
        fix=fix,
        yes=yes,
        fix_rules=fix_rules,
        fix_dry_run=fix_dry_run,
    )
