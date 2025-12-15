from __future__ import annotations

from typing import Any, Optional
from pathlib import Path as _Path
import os

import typer
import click
from typing_extensions import Annotated

from flujo.exceptions import InfiniteRedirectError, PausedException, PipelineAbortSignal
from . import helpers
from .helpers import (
    create_flujo_runner,
    execute_pipeline_with_output_handling,
    print_rich_or_typer,
    update_project_budget,
    find_project_root,
)


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
    # Sync key helpers from flujo.cli.main so test monkeypatches apply
    try:
        import importlib

        _main_mod = importlib.import_module("flujo.cli.main")
        _helpers_mod = importlib.import_module("flujo.cli.helpers")
        for _nm in [
            "load_pipeline_from_yaml_file",
            "create_flujo_runner",
            "execute_pipeline_with_output_handling",
            "validate_yaml_text",
            "print_rich_or_typer",
            "parse_context_data",
        ]:
            if hasattr(_main_mod, _nm):
                globals()[_nm] = getattr(_main_mod, _nm)
            elif hasattr(_helpers_mod, _nm):
                globals()[_nm] = getattr(_helpers_mod, _nm)
    except Exception:
        pass
    runner: Any | None = None
    state_backend: Any | None = None
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
                print_rich_or_typer(
                    "[red]--output-dir is required when running --non-interactive to specify where to write pipeline.yaml[/red]",
                    stderr=True,
                )
                raise typer.Exit(2)

            # Track whether user supplied --goal flag explicitly (HITL skip rule)
            goal_flag_provided = goal is not None

            # Prompt for goal if not provided and interactive
            if goal is None and not non_interactive:
                goal = typer.prompt("What is your goal for this pipeline?")
            if goal is None:
                print_rich_or_typer("[red]--goal is required in --non-interactive mode[/red]")
                raise typer.Exit(2)
            # Prepare initial context data
            from .helpers import parse_context_data

            # Ensure built-in skills are registered and collect available skills
            try:
                import flujo.builtins as _ensure_builtins  # noqa: F401
                from flujo.infra.skill_registry import get_skill_registry_provider as _get_provider

                _reg = _get_provider().get_registry()
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
                try:
                    from . import main as _main_mod

                    main_fn = getattr(_main_mod, "load_pipeline_from_yaml_file", None)
                except Exception:
                    main_fn = None
                helper_fn = getattr(helpers, "load_pipeline_from_yaml_file", None)
                if (
                    main_fn is not None
                    and getattr(main_fn, "__module__", None) != "flujo.cli.helpers"
                ):
                    fn = main_fn  # prefer explicitly monkeypatched main export
                elif helper_fn is not None:
                    fn = helper_fn  # helper may be monkeypatched (tests target this path)
                else:
                    fn = main_fn
                pipeline_obj = None
                if fn is not None:
                    try:
                        # Allow tests to monkeypatch the loader; ignore missing sentinel path
                        pipeline_obj = fn("<injected>")
                    except (typer.Exit, FileNotFoundError):
                        pipeline_obj = None
                    except Exception:
                        # Fall back to programmatic builder on any injected-path failure
                        pipeline_obj = None
                if pipeline_obj is None:
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

                state_backend = _load_backend_from_config()
            except Exception:
                state_backend = None

            runner = create_flujo_runner(
                pipeline=pipeline_obj,
                context_model_class=_ArchitectContext,
                initial_context_data=initial_context_data,
                state_backend=state_backend,
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
                    print_rich_or_typer(
                        f"[grey58]No YAML found. step_history_len={len(sh)}[/grey58]"
                    )
                    try:
                        ctx = getattr(result, "final_pipeline_context", None)
                        if ctx is not None:
                            g = getattr(ctx, "generated_yaml", None)
                            y = getattr(ctx, "yaml_text", None)
                            print_rich_or_typer(
                                f"[grey58]final_ctx has generated_yaml={bool(g)} yaml_text={bool(y)}[/grey58]"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass
                print_rich_or_typer(
                    "[red]Architect did not produce YAML (context.generated_yaml missing)"
                )
                raise typer.Exit(1)

            # Security gating: detect side-effect tools and require confirmation unless explicitly allowed
            from .helpers import find_side_effect_skills_in_yaml, enrich_yaml_with_required_params

            side_effect_skills = find_side_effect_skills_in_yaml(
                yaml_text, base_dir=output_dir or os.getcwd()
            )
            if side_effect_skills and not allow_side_effects:
                print_rich_or_typer(
                    "[red]This blueprint references side-effect skills that may perform external actions:"
                )
                for sid in side_effect_skills:
                    typer.echo(f"  - {sid}")
                if non_interactive:
                    print_rich_or_typer(
                        "[red]Non-interactive mode: re-run with --allow-side-effects to proceed."
                    )
                    # Also emit the flag hint to stdout so Typer tests can assert on it deterministically.
                    typer.echo("--allow-side-effects")
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
                print_rich_or_typer(
                    "[red]No YAML was generated from the architect pipeline (plan rejected or writer failed). Aborting.",
                    stderr=True,
                )
                raise typer.Exit(1)

            # Validate in-memory before writing (allow tests to monkeypatch main.validate_yaml_text)
            try:
                from . import main as _main_mod

                validator = getattr(_main_mod, "validate_yaml_text", helpers.validate_yaml_text)
            except Exception:
                validator = helpers.validate_yaml_text
            report = validator(yaml_text, base_dir=output_dir or os.getcwd())
            if not report.is_valid and strict:
                print_rich_or_typer("[red]Generated YAML is invalid under --strict")
                raise typer.Exit(1)

            # Interactive HITL: show plan and ask for approval when --goal flag was not provided
            if hitl_requested:
                try:
                    preview = yaml_text.strip()
                    # Trim extremely long previews
                    if len(preview) > 2000:
                        preview = preview[:2000] + "\n... (truncated)"
                    print_rich_or_typer("\n[bold]Proposed pipeline plan (YAML preview):[/bold]")
                    typer.echo(preview)
                except Exception:
                    pass
                approved = typer.confirm(
                    "Proceed to generate pipeline from this plan?", default=True
                )
                if not approved:
                    print_rich_or_typer("[red]Creation aborted by user at plan approval stage.")
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
                print_rich_or_typer(
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
            print_rich_or_typer(f"[green]Wrote: {out_yaml}")

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
                            print_rich_or_typer(
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
                            print_rich_or_typer(
                                "[red]Creation aborted by user at budget confirmation stage."
                            )
                            raise typer.Exit(1)
                except Exception:
                    # Fall back to skipping budget confirmation on unexpected prompt failures
                    budget_val = None

            # Optionally update flujo.toml budget
            try:
                effective_budget = budget_val
                if effective_budget is None and budget is not None:
                    try:
                        effective_budget = float(budget)
                    except Exception:
                        effective_budget = None
                if effective_budget is not None:
                    pipeline_name = (
                        name or _extract_pipeline_name_from_yaml(yaml_text) or "pipeline"
                    )
                    flujo_toml_path = _Path(out_dir) / "flujo.toml"
                    if flujo_toml_path.exists():
                        update_project_budget(
                            flujo_toml_path, pipeline_name, float(effective_budget)
                        )
                        print_rich_or_typer(
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
                            from flujo.utils.async_bridge import run_sync

                            run_sync(close_async_fn())
            except Exception:
                pass
    except typer.Exit:
        # Preserve explicit exit codes for wizard/architect flows without wrapping
        raise
    except (PausedException, PipelineAbortSignal, InfiniteRedirectError):
        # Allow orchestration control-flow signals to propagate
        raise
    except Exception as e:
        print_rich_or_typer(f"[red]Failed to create pipeline: {e}", stderr=True)
        raise typer.Exit(1) from e


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
    map_step = (map_step_name or "process").strip() or "process"

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
            f"        - kind: step\n          name: {map_step}\n          updates_context: false"
        )
        lines.append("      init:")
        lines.append(
            '        - set: "context.import_artifacts.extras.note"\n          value: "mapping"'
        )
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


__all__ = ["create", "_run_create_wizard"]
