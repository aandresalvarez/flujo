from __future__ import annotations
# NOTE: This module contains experimental/dev-only CLI commands and is not yet
# fully typed under mypy --strict.
# mypy: ignore-errors

from pathlib import Path
from typing import Any, Optional

import json
import os
import keyword
import typer
import click
from typing_extensions import Annotated

from flujo.infra import telemetry
from flujo.domain.dsl import Pipeline
from flujo.cli.generators.openapi_agent_generator import (
    generate_openapi_agents,
    load_openapi_spec,
)
from .helpers import (
    get_masked_settings_dict,
    get_pipeline_explanation,
    get_version_string,
    find_project_root,
    load_mermaid_code,
    load_pipeline_from_file,
    print_rich_or_typer,
    validate_pipeline_file,
)

logfire = telemetry.logfire


def gen_context(
    pipeline_file: Annotated[
        Optional[str],
        typer.Argument(
            help="Path to pipeline.yaml (defaults to project pipeline.yaml if present)",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output path for generated context model"),
    ] = "context.py",
    model_name: Annotated[
        str,
        typer.Option("--model-name", "-m", help="Name of generated Pydantic model"),
    ] = "GeneratedContext",
) -> None:
    """
    Generate a Pydantic context model by scanning pipeline YAML for context references
    and declared input/output keys.
    """
    import re
    import yaml

    try:
        if pipeline_file is None:
            root = find_project_root()
            candidate = (Path(root) / "pipeline.yaml").resolve()
            if not candidate.exists():
                raise FileNotFoundError("pipeline.yaml not found; pass path explicitly")
            pipeline_file = str(candidate)

        text = Path(pipeline_file).read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    except Exception as e:  # noqa: BLE001
        print_rich_or_typer(f"[red]Failed to load pipeline: {e}", stderr=True)
        raise typer.Exit(1) from e

    keys: set[str] = set()

    for match in re.findall(r"context\\.([A-Za-z0-9_\\.]+)", text):
        root = match.split(".", 1)[0]
        if root:
            keys.add(root)

    def _collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"input_keys", "output_keys"} and isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            keys.add(item.split(".", 1)[0])
                else:
                    _collect(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect(item)

    _collect(data)

    if not keys:
        print_rich_or_typer("[yellow]No context fields detected; generated empty model.[/yellow]")

    lines = [
        "from __future__ import annotations",
        "from typing import Any, Optional",
        "from pydantic import BaseModel",
        "",
        f"class {model_name}(BaseModel):",
    ]
    if not keys:
        lines.append("    pass")
    else:
        for key in sorted(keys):
            lines.append(f"    {key}: Optional[Any] = None")

    try:
        Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print_rich_or_typer(f"[green]Generated context model at {output}[/green]")
    except Exception as e:  # noqa: BLE001
        print_rich_or_typer(f"[red]Failed to write context model: {e}", stderr=True)
        raise typer.Exit(1) from e


def import_openapi(
    spec: Annotated[str, typer.Argument(help="Path or URL to OpenAPI/Swagger spec")],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output directory for generated models (a Python package). "
                "Models are written to <output>/generated_models.py."
            ),
        ),
    ] = "generated_tools",
    target_python_version: Annotated[
        str, typer.Option("--python-version", help="Target Python version (e.g., 3.13)")
    ] = "3.13",
    base_class: Annotated[
        str, typer.Option("--base-class", help="Base class for models (pydantic style)")
    ] = "pydantic.BaseModel",
    disable_timestamp: Annotated[
        bool,
        typer.Option("--disable-timestamp/--enable-timestamp", help="Toggle generated timestamp"),
    ] = True,
    generate_agents: Annotated[
        bool,
        typer.Option(
            "--generate-agents/--skip-agents", help="Generate agent wrappers alongside models"
        ),
    ] = True,
    agents_filename: Annotated[
        str,
        typer.Option(
            "--agents-filename",
            help="Filename for generated agent wrappers (relative to output dir)",
        ),
    ] = "openapi_agents.py",
) -> None:
    """
    Generate Pydantic models/tools from an OpenAPI spec using datamodel-code-generator.

    Requires `datamodel-code-generator` to be installed (pip/uv). Emits a friendly
    error if the dependency is missing.
    """
    output_dir = Path(output)
    if not output_dir.name.isidentifier() or keyword.iskeyword(output_dir.name):
        print_rich_or_typer(
            (
                "[red]Output directory name must be a valid importable Python package name.[/red]\n"
                f"Got: {output_dir.name!r}. Choose something like `generated_tools`."
            ),
            stderr=True,
        )
        raise typer.Exit(1)

    try:
        from datamodel_code_generator import main as dcg_main  # type: ignore
    except Exception:  # pragma: no cover - dependency not installed
        print_rich_or_typer(
            "[red]datamodel-code-generator is not installed. "
            "Install with `uv add datamodel-code-generator` or "
            "`pip install datamodel-code-generator`.",
            stderr=True,
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_file = output_dir / "generated_models.py"

    args = [
        "--input",
        spec,
        "--input-file-type",
        "openapi",
        "--output",
        str(models_file),
        "--target-python-version",
        target_python_version,
    ]
    if disable_timestamp:
        args.append("--disable-timestamp")
    if base_class:
        args.extend(["--base-class", base_class])

    logfire.info(
        "[OpenAPI] Generating models",
        extra={"spec": spec, "output": output, "target_python_version": target_python_version},
    )
    try:
        dcg_main(args)
    except SystemExit as exc:  # datamodel-code-generator uses sys.exit
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            print_rich_or_typer(f"[red]datamodel-code-generator failed (exit={code})", stderr=True)
            raise typer.Exit(code)
    except Exception as exc:  # pragma: no cover - passthrough errors
        print_rich_or_typer(f"[red]datamodel-code-generator failed: {exc}", stderr=True)
        raise typer.Exit(1) from exc

    if generate_agents:
        try:
            spec_dict = load_openapi_spec(spec)
            models_package = output_dir.name.replace("-", "_")
            agents_path = generate_openapi_agents(
                spec=spec_dict,
                models_package=models_package,
                output_dir=output_dir,
                agents_filename=agents_filename,
                models_module=models_file.stem,
            )
            logfire.info(
                "[OpenAPI] Generated agent wrappers",
                extra={"agents_path": str(agents_path), "models_package": models_package},
            )
        except Exception as exc:  # pragma: no cover - passthrough errors
            print_rich_or_typer(
                f"[red]Failed to generate OpenAPI agent wrappers: {exc}", stderr=True
            )
            raise typer.Exit(1) from exc

    init_path = output_dir / "__init__.py"
    init_lines: list[str] = [
        '"""Auto-generated package from `flujo dev import-openapi`."""',
        "from __future__ import annotations",
        "",
        "from .generated_models import *  # noqa: F401,F403",
    ]
    if generate_agents:
        init_lines.extend(
            [
                "",
                f"from .{Path(agents_filename).stem} import (",
                "    OPERATION_FUNCS,",
                "    RESPONSE_MODEL_NAMES,",
                "    make_openapi_agent,",
                "    make_openapi_operation_agent,",
                ")",
            ]
        )
    init_path.write_text("\n".join(init_lines) + "\n", encoding="utf-8")


logfire = telemetry.logfire


def version_cmd() -> None:
    """Print the package version."""
    version = get_version_string()
    typer.echo(f"flujo version: {version}")


def show_config_cmd() -> None:
    """
    Print effective Settings with secrets masked.

    Returns:
        None: Prints configuration to stdout
    """
    typer.echo(get_masked_settings_dict())


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
                        import tomli as _tomllib  # type: ignore[no-redef]
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
                        import tomli as _tomllib  # type: ignore[no-redef]
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
                    prev_err = [x for x in (prev_raw.get("errors") or []) if isinstance(x, dict)]
                    prev_warn = [x for x in (prev_raw.get("warnings") or []) if isinstance(x, dict)]
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
                **({"fixes": applied_fixes_metrics} if applied_fixes_metrics is not None else {}),
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
        print_rich_or_typer(f"[red]Validation failed: {type(e).__name__}: {e}[/red]", stderr=True)
        if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
            typer.echo("\nTraceback:", err=True)
            typer.echo("".join(_tb.format_exception(e)), err=True)
        raise typer.Exit(EX_RUNTIME_ERROR) from e


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
        typer.Option("--baseline", help="Path to a previous JSON report to compute deltas against"),
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


__all__ = ["register_dev_commands"]


def register_dev_commands(dev_app: typer.Typer) -> None:
    dev_app.command(name="version")(version_cmd)
    dev_app.command(name="show-config")(show_config_cmd)
    dev_app.command(name="explain")(explain)
    dev_app.command(name="validate")(validate_dev)
    dev_app.command(name="compile-yaml")(compile)
    dev_app.command(name="visualize")(pipeline_mermaid_cmd)
    dev_app.command(name="import-openapi")(import_openapi)
    dev_app.command(name="gen-context")(gen_context)
