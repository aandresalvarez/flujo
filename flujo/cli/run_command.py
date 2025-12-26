"""Run command implementation for Flujo CLI."""

from __future__ import annotations


from typing import Any, Optional
import typer
import click
import json
from pathlib import Path

from flujo.exceptions import UsageLimitExceededError
from flujo.infra import telemetry
import flujo.builtins as _flujo_builtins  # noqa: F401 - register builtins
from flujo.utils.async_bridge import run_sync
from .helpers import (
    setup_run_command_environment,
    load_pipeline_from_yaml_file,
    create_flujo_runner,
    execute_pipeline_with_output_handling,
    display_pipeline_results,
    apply_cli_defaults,
    parse_context_data,
    find_project_root,
    print_rich_or_typer,
)

logfire = telemetry.logfire


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
    live: bool = typer.Option(
        False,
        "--live",
        "--progress",
        help="Show live step progress (compact panels)",
    ),
    # Visualization controls (defaults preserve current behavior)
    summary: bool = typer.Option(
        False, "--summary", help="Only show final output, totals and run id"
    ),
    show_context: bool = typer.Option(
        True,
        "--show-context/--no-context",
        help="Include or hide the final context block",
    ),
    show_steps: bool = typer.Option(
        True, "--show-steps/--no-steps", help="Include or hide the step results table"
    ),
    show_output_column: bool = typer.Option(
        True,
        "--show-output-column/--no-output-column",
        help="Include or hide the 'Output' column in the step table",
    ),
    output_preview_len: Optional[int] = typer.Option(
        None,
        "--output-preview-len",
        help="Max chars for outputs in the step table (default 100)",
    ),
    final_output_format: Optional[str] = typer.Option(
        None,
        "--final-output-format",
        help="How to render final output (auto, raw, json, yaml, md)",
        click_type=click.Choice(["auto", "raw", "json", "yaml", "md"], case_sensitive=False),
    ),
    pager: bool = typer.Option(False, "--pager", help="Page long output with a scrollable viewer"),
    only_steps: Optional[str] = typer.Option(
        None,
        "--only-steps",
        help="Comma-separated list of step names to include in the step table",
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
    state_backend: Any | None = None
    try:
        # Apply CLI defaults from configuration file
        cli_args = apply_cli_defaults(
            "run",
            pipeline_name=pipeline_name,
            json_output=json_output,
        )
        pipeline_name_val = cli_args.get("pipeline_name")
        if isinstance(pipeline_name_val, str):
            pipeline_name = pipeline_name_val
        json_output_val = cli_args.get("json_output")
        if isinstance(json_output_val, bool):
            json_output = json_output_val

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

            # CRITICAL FIX: Ensure initial_prompt is set in context for YAML files
            # This enables ask_user and other skills to access piped input
            if initial_context_data is None:
                initial_context_data = {}
            if "initial_prompt" not in initial_context_data:
                initial_context_data["initial_prompt"] = input_data
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
                print_rich_or_typer(f"[red]Validation crashed: {ve}", stderr=True)
                raise typer.Exit(1) from ve

            if not validation_report.is_valid:
                print_rich_or_typer("[red]Pipeline validation failed before run:[/red]")
                for f in validation_report.errors:
                    loc = f"{f.step_name}: " if f.step_name else ""
                    if f.suggestion:
                        typer.echo(
                            f"- [{f.rule_id}] {loc}{f.message} -> Suggestion: {f.suggestion}"
                        )
                    else:
                        typer.echo(f"- [{f.rule_id}] {loc}{f.message}")
                raise typer.Exit(1)

        # Zero-config UX before dry-run short-circuit: warn/offer init once interactively
        try:
            import sys as _sys
            from ..infra.config_manager import ConfigManager as _CfgMgr
            from ..infra.settings import get_settings as _get_settings

            _cfg = _CfgMgr()
            _has_config = _cfg.config_path is not None
            _is_tty = bool(getattr(_sys.stdin, "isatty", lambda: False)())
            _test_mode = bool(getattr(_get_settings(), "test_mode", False))
            if (not _has_config) and _is_tty and not json_output and not _test_mode:
                try:
                    from typer import secho as _se, confirm as _confirm

                    _se(
                        "No flujo.toml found in this directory. Running in zero-config mode.",
                        fg="yellow",
                    )
                    if _confirm("Create minimal flujo.toml now?", default=True):
                        _path = Path.cwd() / "flujo.toml"
                        if not _path.exists():
                            _path.write_text(
                                """
# Minimal flujo.toml (generated by `flujo run`)

state_uri = "sqlite:///flujo_ops.db"
# env_file = ".env"
                                """.strip(),
                                encoding="utf-8",
                            )
                            _se("Created flujo.toml (sqlite:///flujo_ops.db)", fg="green")
                except Exception:
                    pass
        except Exception:
            pass

        # If dry-run requested, stop after validation
        if dry_run:
            try:
                names = [s.name for s in getattr(pipeline_obj, "steps", [])]
            except Exception:
                names = []
            if json_output:
                typer.echo(json.dumps({"validated": True, "steps": names}))
            else:
                print_rich_or_typer("[green]Pipeline parsed and validated (dry run)")
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
        try:
            from .config import load_backend_from_config as _load_backend_from_config

            state_backend = _load_backend_from_config()
        except Exception:
            state_backend = None

        # Zero-config UX: if no flujo.toml was found, warn clearly and optionally offer init.
        try:
            import sys as _sys
            from ..infra.config_manager import ConfigManager as _CfgMgr
            from ..infra.settings import get_settings as _get_settings
            from ..utils.model_utils import extract_model_id, extract_provider_and_model
            from ..domain.dsl.step import Step as _Step
            from ..domain.dsl.step import HumanInTheLoopStep as _Hitl

            _cfg = _CfgMgr()
            _has_config = _cfg.config_path is not None
            _is_tty = bool(getattr(_sys.stdin, "isatty", lambda: False)())
            _test_mode = bool(getattr(_get_settings(), "test_mode", False))

            def _secho(msg: str, fg: str = "yellow") -> None:
                try:
                    from typer import secho as _se

                    _se(msg, fg=fg)
                except Exception as e:  # noqa: BLE001, S110
                    try:
                        logfire.debug(f"secho failed: {type(e).__name__}: {e}")
                    except Exception:
                        pass

            if not _has_config:
                # One-time prominent warning in interactive mode; telemetry will also capture.
                if _is_tty and not json_output and not _test_mode:
                    _secho(
                        "No flujo.toml found in this directory. Running in zero-config mode.",
                        fg="yellow",
                    )
                    # Durability warning for HITL when state backend is ephemeral
                    try:
                        has_hitl = any(
                            isinstance(s, _Hitl) for s in getattr(pipeline_obj, "steps", []) or []
                        )
                    except Exception:
                        has_hitl = False
                    # Consider both backend type and configured state URI for durability
                    try:
                        from ..infra.config_manager import get_state_uri as _get_state_uri

                        _uri = _get_state_uri(force_reload=True)
                    except Exception:  # noqa: BLE001
                        _uri = None
                    _ephemeral = (
                        not _uri
                        or str(_uri).strip().lower()
                        in {"memory", "memory://", "mem://", "inmemory://"}
                        or getattr(state_backend, "__class__", object).__name__ == "InMemoryBackend"
                    )
                    if has_hitl and _ephemeral:
                        _secho(
                            "HITL detected and using in-memory state; pauses won't survive restarts.",
                            fg="yellow",
                        )
                        _secho(
                            "Tip: run 'flujo init' or set FLUJO_STATE_URI=sqlite:///flujo_ops.db",
                            fg="yellow",
                        )

                    # Credential hints: scan agent models for provider and suggest env var
                    try:
                        providers = set()
                        for st in getattr(pipeline_obj, "steps", []) or []:
                            if not isinstance(st, _Step):
                                continue
                            agent = getattr(st, "agent", None)
                            mid = (
                                extract_model_id(agent, step_name=getattr(st, "name", "step"))
                                if agent
                                else None
                            )
                            if mid:
                                prov, _ = extract_provider_and_model(mid)
                                if prov:
                                    providers.add(prov)
                        import os as _os2

                        missing = []
                        if "openai" in providers and not _os2.getenv("OPENAI_API_KEY"):
                            missing.append("OPENAI_API_KEY")
                        if "anthropic" in providers and not _os2.getenv("ANTHROPIC_API_KEY"):
                            missing.append("ANTHROPIC_API_KEY")
                        if "google" in providers and not _os2.getenv("GOOGLE_API_KEY"):
                            missing.append("GOOGLE_API_KEY")
                        if "cohere" in providers and not _os2.getenv("COHERE_API_KEY"):
                            missing.append("COHERE_API_KEY")
                        if missing:
                            _secho(
                                "Missing API keys: "
                                + ", ".join(missing)
                                + ". Set env vars or add env_file in flujo.toml.",
                                fg="yellow",
                            )
                    except Exception:
                        pass

                    # Offer quick init
                    try:
                        from typer import confirm as _confirm

                        if _confirm("Create minimal flujo.toml now?", default=True):
                            _path = Path.cwd() / "flujo.toml"
                            if not _path.exists():
                                _path.write_text(
                                    """
# Minimal flujo.toml (generated by `flujo run`)

# Default local state for pause/resume durability
state_uri = "sqlite:///flujo_ops.db"

# Optional: load environment variables (e.g., API keys) from a file
# env_file = ".env"

# Budgets and pricing (optional)
[budgets]
# default.total_cost_usd_limit = 10.0
# default.total_tokens_limit = 100000
                                    """.strip(),
                                    encoding="utf-8",
                                )
                                _secho("Created flujo.toml (sqlite:///flujo_ops.db)", fg="green")
                            else:
                                _secho("flujo.toml already exists; skipping creation", fg="yellow")
                    except Exception:
                        pass
                else:
                    # Non-interactive/JSON/test: log a concise hint (stdout friendly)
                    try:
                        from ..infra import telemetry as _tele

                        _tele.logfire.warning("No flujo.toml found; zero-config defaults in effect")
                    except Exception:
                        pass
        except Exception as e:  # noqa: BLE001, S110
            # Never block execution due to UX helpers; keep it observable under --debug
            try:
                logfire.debug(f"zero-config UX guard suppressed: {type(e).__name__}: {e}")
            except Exception:
                pass

        runner = create_flujo_runner(
            pipeline=pipeline_obj,
            context_model_class=context_model_class,
            initial_context_data=initial_context_data,
            state_backend=state_backend,
            debug=debug,
            live=live,
        )

        # Execute pipeline using helper function
        if json_output:
            result_json = execute_pipeline_with_output_handling(
                runner=runner,
                input_data=input_data,
                run_id=run_id,
                json_output=True,
            )
            typer.echo(result_json)
            return

        result = execute_pipeline_with_output_handling(
            runner=runner,
            input_data=input_data,
            run_id=run_id,
            json_output=False,
        )

        # If execution yielded failures, print a concise credentials hint when likely.
        try:

            def _collect_failed_feedback(res: Any) -> list[str]:
                vals: list[str] = []
                try:
                    for sr in getattr(res, "step_history", []) or []:
                        if not getattr(sr, "success", True):
                            fb = getattr(sr, "feedback", None)
                            if isinstance(fb, str) and fb:
                                vals.append(fb)
                        # Inspect nested histories if present
                        nested = getattr(sr, "step_history", None)
                        if isinstance(nested, list) and nested:
                            vals.extend(
                                [
                                    str(getattr(ns, "feedback", ""))
                                    for ns in nested
                                    if not getattr(ns, "success", True)
                                    and isinstance(getattr(ns, "feedback", None), str)
                                ]
                            )
                except Exception:
                    pass
                return vals

            def _likely_missing_credentials(text: str) -> bool:
                t = text.lower()
                # Broad auth/credentials patterns commonly seen across SDKs and HTTP layers
                tokens = [
                    "api key",
                    "no api key",
                    "missing api key",
                    "authenticationerror",
                    "unauthorized",
                    "forbidden",
                    "invalid api key",
                    "missing required",
                    "credentials",
                    "bearer token",
                    "401",
                    "403",
                ]
                return any(tok in t for tok in tokens)

            def _print_credentials_hint(providers: set[str]) -> None:
                import os as _os3

                missing: list[str] = []
                # Only suggest envs that are actually unset to avoid noise
                if (not providers or "openai" in providers) and not _os3.getenv("OPENAI_API_KEY"):
                    missing.append("OPENAI_API_KEY")
                if (not providers or "anthropic" in providers) and not _os3.getenv(
                    "ANTHROPIC_API_KEY"
                ):
                    missing.append("ANTHROPIC_API_KEY")
                if (not providers or "google" in providers) and not _os3.getenv("GOOGLE_API_KEY"):
                    missing.append("GOOGLE_API_KEY")
                if (not providers or "cohere" in providers) and not _os3.getenv("COHERE_API_KEY"):
                    missing.append("COHERE_API_KEY")
                if missing and not json_output:
                    try:
                        from typer import secho as _se

                        _se(
                            "Credentials hint: set "
                            + ", ".join(missing)
                            + " or add an env_file in flujo.toml.",
                            fg="yellow",
                        )
                    except Exception:
                        pass

            # Build provider hints from the pipeline’s declared agents
            providers_hint: set[str] = set()
            try:
                from ..utils.model_utils import extract_model_id, extract_provider_and_model
                from ..domain.dsl.step import Step as _Step

                for st in getattr(pipeline_obj, "steps", []) or []:
                    if not isinstance(st, _Step):
                        continue
                    agent = getattr(st, "agent", None)
                    mid = (
                        extract_model_id(agent, step_name=getattr(st, "name", "step"))
                        if agent
                        else None
                    )
                    if mid:
                        prov, _ = extract_provider_and_model(mid)
                        if prov:
                            providers_hint.add(prov)
            except Exception:
                providers_hint = set()

            # Check any failed step feedbacks
            for fb in _collect_failed_feedback(result):
                if _likely_missing_credentials(fb):
                    _print_credentials_hint(providers_hint)
                    break
        except Exception:
            # Never block execution due to hinting
            pass

        # Interactive HITL resume loop: if paused and in TTY, prompt and resume
        if not json_output:
            try:
                import sys as _sys

                def _is_paused(_res: Any) -> tuple[bool, str | None]:
                    try:
                        ctx = getattr(_res, "final_pipeline_context", None)
                        if ctx is not None and getattr(ctx, "status", None) == "paused":
                            return True, getattr(ctx, "pause_message", None)
                    except Exception:
                        pass
                    return False, None

                paused, msg = _is_paused(result)
                while paused and _sys.stdin.isatty():
                    prompt_msg = msg or "Provide input to resume:"
                    human = typer.prompt(prompt_msg)
                    # Resume via runner
                    result = run_sync(runner.resume_async(result, human))
                    paused, msg = _is_paused(result)
            except Exception:
                # If resume fails, fall through to normal display (will show paused message)
                pass

        # Normalize visualization flags
        eff_show_steps = show_steps
        eff_show_context = show_context
        if summary:
            eff_show_steps = False
            eff_show_context = False
        include_list = None
        try:
            if only_steps and only_steps.strip():
                include_list = [s.strip() for s in only_steps.split(",") if s.strip()]
        except Exception:
            include_list = None

        display_pipeline_results(
            result,
            run_id,
            json_output,
            show_steps=eff_show_steps,
            show_context=eff_show_context,
            show_output_column=show_output_column,
            output_preview_len=output_preview_len,
            final_output_format=(final_output_format or "auto"),
            pager=pager,
            only_steps=include_list,
        )
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
                            if k.startswith("flujo.") or k in ("success", "latency_s", "step_input")
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
                from datetime import datetime as _dt, UTC as _UTC

                root = find_project_root()
                base_dir = _Path(root) if root else _Path.cwd()
                debug_dir = base_dir / "debug"
                ts = _dt.now(_UTC).strftime("%Y%m%d_%H%M%S")
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
                from datetime import datetime as _dt, UTC as _UTC
                import os as _os
                from pydantic import BaseModel as _BM
                import dataclasses as _dc

                export_path_obj = _Path(export_path).expanduser().resolve()
                export_path_obj.parent.mkdir(parents=True, exist_ok=True)

                def _serialize_obj(obj: Any) -> Any:
                    if obj is None:
                        return None
                    if isinstance(obj, (str, int, float, bool)):
                        return obj

                    try:
                        if isinstance(obj, _BM):
                            try:
                                return obj.model_dump(mode="json")
                            except TypeError:
                                return obj.model_dump()
                    except Exception:
                        pass

                    try:
                        if _dc.is_dataclass(obj) and not isinstance(obj, type):
                            return _dc.asdict(obj)
                    except Exception:
                        pass

                    if isinstance(obj, list):
                        return [_serialize_obj(item) for item in obj]

                    if isinstance(obj, dict):
                        return {str(k): _serialize_obj(v) for k, v in obj.items()}

                    if hasattr(obj, "__dict__"):
                        return {
                            str(k): _serialize_obj(v)
                            for k, v in obj.__dict__.items()
                            if not k.startswith("_")
                        }

                    return str(obj)

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
                        "output": _serialize_obj(getattr(sr, "output", None)),
                        "metadata": getattr(sr, "metadata_", {}) or {},
                        "step_history": nested,
                    }

                def _context_to_dict(ctx: Any) -> dict[str, Any]:
                    if ctx is None:
                        return {}
                    try:
                        if hasattr(ctx, "model_dump"):
                            try:
                                base = ctx.model_dump(mode="json")
                            except TypeError:
                                base = ctx.model_dump()
                        else:
                            base = _serialize_obj(ctx)
                    except Exception:
                        base = {}
                    # Augment with common transient fields for debugging
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
                        base["hitl_history"] = _serialize_obj(getattr(ctx, "hitl_history", []))
                    except Exception:
                        pass
                    try:
                        base["command_log"] = _serialize_obj(getattr(ctx, "command_log", []))
                    except Exception:
                        pass
                    return base if isinstance(base, dict) else {"context": base}

                # Final payload
                exported_at = _dt.now(_UTC).isoformat().replace("+00:00", "Z")
                payload = {
                    "exported_at": exported_at,
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
                    print_rich_or_typer(f"[green]Wrote full debug log to[/green] {export_path_obj}")
            except Exception as e:
                if not json_output:
                    print_rich_or_typer(f"[red]Failed to export debug log:[/red] {e}", stderr=True)

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
            print_rich_or_typer(f"[red]Budget exceeded: {e}[/red]", stderr=True)
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

        print_rich_or_typer(f"[red]Error running pipeline: {type(e).__name__}: {e}", stderr=True)
        # If exception looks like missing credentials, print one-line hint
        try:
            msg = str(e)

            def _looks_like_creds_err(t: str) -> bool:
                t = t.lower()
                return any(
                    s in t
                    for s in [
                        "api key",
                        "unauthorized",
                        "authentication",
                        "invalid key",
                        "401",
                        "403",
                    ]
                )

            if _looks_like_creds_err(msg) and not json_output:
                # Reuse providers hint from pipeline where available
                providers_hint_err: set[str] = set()
                try:
                    from ..utils.model_utils import extract_model_id, extract_provider_and_model
                    from ..domain.dsl.step import Step as _Step

                    for st in getattr(pipeline_obj, "steps", []) or []:
                        if not isinstance(st, _Step):
                            continue
                        agent = getattr(st, "agent", None)
                        mid = (
                            extract_model_id(agent, step_name=getattr(st, "name", "step"))
                            if agent
                            else None
                        )
                        if mid:
                            prov, _ = extract_provider_and_model(mid)
                            if prov:
                                providers_hint_err.add(prov)
                except Exception:
                    providers_hint_err = set()
                # Print concise hint with specific envs when possible
                import os as _os4

                missing_envs = []
                if (not providers_hint_err or "openai" in providers_hint_err) and not _os4.getenv(
                    "OPENAI_API_KEY"
                ):
                    missing_envs.append("OPENAI_API_KEY")
                if (
                    not providers_hint_err or "anthropic" in providers_hint_err
                ) and not _os4.getenv("ANTHROPIC_API_KEY"):
                    missing_envs.append("ANTHROPIC_API_KEY")
                if (not providers_hint_err or "google" in providers_hint_err) and not _os4.getenv(
                    "GOOGLE_API_KEY"
                ):
                    missing_envs.append("GOOGLE_API_KEY")
                if (not providers_hint_err or "cohere" in providers_hint_err) and not _os4.getenv(
                    "COHERE_API_KEY"
                ):
                    missing_envs.append("COHERE_API_KEY")
                if missing_envs:
                    try:
                        from typer import secho as _se

                        _se(
                            "Credentials hint: set "
                            + ", ".join(missing_envs)
                            + " or add an env_file in flujo.toml.",
                            fg="yellow",
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
            typer.echo("\nTraceback:", err=True)
            typer.echo("".join(_tb.format_exception(e)), err=True)
        from .exit_codes import EX_RUNTIME_ERROR

        raise typer.Exit(EX_RUNTIME_ERROR)
    finally:
        try:
            if runner is not None:
                close_fn = getattr(runner, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass

        try:
            if state_backend is not None:
                close_sync_fn = getattr(state_backend, "close_sync", None)
                if callable(close_sync_fn):
                    close_sync_fn()
                else:
                    close_async_fn = getattr(state_backend, "close", None)
                    if callable(close_async_fn):
                        run_sync(close_async_fn())
        except Exception:
            pass

        # Tests may invoke multiple CLI commands in the same process; reset the global skill registry
        # provider between runs to prevent cross-test pollution from dynamically loaded workflows.
        try:
            from flujo.infra.skill_registry import reset_skill_registry_provider

            reset_skill_registry_provider()
        except Exception:
            pass
