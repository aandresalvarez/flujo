"""CLI entry point for flujo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast
import typer
import click
import json
from pathlib import Path
from importlib import resources as importlib_resources
from flujo.infra.config_manager import get_cli_defaults as _get_cli_defaults
from flujo.exceptions import ConfigurationError, SettingsError
from flujo.infra import telemetry
import flujo.builtins as _flujo_builtins  # noqa: F401  # Register builtin skills on CLI import
from typing_extensions import Annotated
from rich.console import Console
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
    validate_yaml_text,
    parse_context_data,
    load_pipeline_from_file,
    find_project_root,
    scaffold_project,
    update_project_budget,
)
import click.testing
import os

# Expose Flujo class for tests that monkeypatch flujo.cli.main.Flujo.run
from flujo.application.runner import Flujo as _Flujo  # re-export for test monkeypatch compatibility

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

            # Assign property at runtime; typing not enforced here
            click.testing.Result.stderr = property(_stderr)  # type: ignore[assignment]
            setattr(click.testing.Result, "_flujo_stderr_shim", True)
    except Exception:
        pass

# Type definitions for CLI
WeightsType = List[Dict[str, Union[str, float]]]
MetadataType = Dict[str, Any]
ScorerType = (
    str  # Changed from Literal["ratio", "weighted", "reward"] to str for typer compatibility
)


app: typer.Typer = typer.Typer(
    rich_markup_mode="markdown",
    help=("A project-based server for building, running, and managing AI workflows."),
)

# Initialize telemetry at the start of CLI execution
telemetry.init_telemetry()
logfire = telemetry.logfire

"""Top-level sub-apps and groups."""
# Top-level: lens remains as its own sub-app
app.add_typer(lens_app, name="lens")

# New developer sub-app and nested experimental group
dev_app: typer.Typer = typer.Typer(help="🛠️  Access advanced developer and diagnostic tools.")
experimental_app: typer.Typer = typer.Typer(help="(Advanced) Experimental and diagnostic commands.")
dev_app.add_typer(experimental_app, name="experimental")

# Budgets live under the dev group
budgets_app: typer.Typer = typer.Typer(help="Budget governance commands")
dev_app.add_typer(budgets_app, name="budgets")

# Register developer app at top level
app.add_typer(dev_app, name="dev")


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


@app.command(
    help=(
        "✨ Initialize a new Flujo workflow project in this directory.\n\n"
        "Use --force to re-initialize templates in an existing project, and --yes to skip confirmation."
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


@dev_app.command(name="validate")
def validate(
    path: Optional[str] = typer.Argument(
        None,
        help="Path to pipeline file. If omitted, uses project pipeline.yaml",
    ),
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit with non-zero status if validation errors are found.",
        ),
    ] = False,
) -> None:
    """Validate a pipeline defined in a file."""
    try:
        if path is None:
            root = find_project_root()
            path = str((Path(root) / "pipeline.yaml").resolve())
        report = validate_pipeline_file(path)
        if report.errors:
            typer.echo("[red]Validation errors detected:")
            typer.echo(
                "[red]See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/"
            )
            for f in report.errors:
                loc = f"{f.step_name}: " if f.step_name else ""
                if f.suggestion:
                    typer.echo(f"- [{f.rule_id}] {loc}{f.message} -> Suggestion: {f.suggestion}")
                else:
                    typer.echo(f"- [{f.rule_id}] {loc}{f.message}")
        if report.warnings:
            typer.echo("[yellow]Warnings:")
            typer.echo(
                "[yellow]See docs: https://aandresalvarez.github.io/flujo/reference/validation_rules/"
            )
            for f in report.warnings:
                loc = f"{f.step_name}: " if f.step_name else ""
                if f.suggestion:
                    typer.echo(f"- [{f.rule_id}] {loc}{f.message} -> Suggestion: {f.suggestion}")
                else:
                    typer.echo(f"- [{f.rule_id}] {loc}{f.message}")
        if report.is_valid:
            typer.echo("[green]Pipeline is valid")
        if strict and not report.is_valid:
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[red]Failed to load pipeline file: {e}", err=True)
        raise typer.Exit(1)


@app.command(help=("🤖 Start a conversation with the AI Architect to build your workflow."))
def create(
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
) -> None:
    """Conversational pipeline generation via the Architect pipeline.

    Loads the bundled architect YAML, runs it with the provided goal, and writes outputs.

    Tip: Using GPT-5? To tune agent timeouts/retries for complex reasoning, see
    docs/guides/gpt5_architect.md (agent-level `timeout`/`max_retries`) and
    step-level `config.timeout` (alias to `timeout_s`) for plugin/validator phases.
    """
    try:
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
            # Suppress specific runner warnings for a clean UX
            try:
                _warnings.filterwarnings("ignore", message="pipeline_name was not provided.*")
                _warnings.filterwarnings("ignore", message="pipeline_id was not provided.*")
            except Exception:
                pass

        try:
            # Prompt for goal if not provided and interactive
            if goal is None and not non_interactive:
                goal = typer.prompt("What is your goal for this pipeline?")
            if goal is None:
                typer.echo("[red]--goal is required in --non-interactive mode[/red]")
                raise typer.Exit(2)
            # Locate bundled architect YAML from package resources
            try:
                with importlib_resources.as_file(
                    importlib_resources.files("flujo.recipes").joinpath("architect_pipeline.yaml")
                ) as p:
                    architect_yaml = str(p)
            except (FileNotFoundError, ModuleNotFoundError):
                typer.echo(
                    "[red]Architect pipeline blueprint not found within the application package.",
                    err=True,
                )
                raise typer.Exit(1)

            # Load architect pipeline
            pipeline_obj = load_pipeline_from_yaml_file(architect_yaml)

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

            initial_context_data = {"user_goal": goal, "available_skills": _available_skills}
            extra_ctx = parse_context_data(None, context_file)
            if isinstance(extra_ctx, dict):
                initial_context_data.update(extra_ctx)
            # Ensure required field for custom context model
            if "initial_prompt" not in initial_context_data:
                initial_context_data["initial_prompt"] = goal

            # Create runner and execute
            # Use a minimal context model for the Architect to allow prepared list
            from pydantic import Field as _Field
            from flujo.domain.models import PipelineContext as _PipelineContext

            class _ArchitectContext(_PipelineContext):
                prepared_steps_for_mapping: list[dict[str, Any]] = _Field(default_factory=list)
                generated_yaml: Optional[str] = None
                available_skills: list[dict[str, Any]] = _Field(default_factory=list)

            runner = create_flujo_runner(
                pipeline=pipeline_obj,
                context_model_class=_ArchitectContext,
                initial_context_data=initial_context_data,
            )

            # For now, require goal as input too (can be refined by architect design)
            result = execute_pipeline_with_output_handling(
                runner=runner, input_data=goal, run_id=None, json_output=False
            )

            # Extract YAML text preferring the most recent step output (repairs), then context
            yaml_text: Optional[str] = None
            try:
                # Prefer latest step output for repaired YAML
                for sr in getattr(result, "step_history", [])[::-1]:
                    out = getattr(sr, "output", None)
                    try:
                        if isinstance(out, dict):
                            if out.get("generated_yaml"):
                                yaml_text = str(out.get("generated_yaml"))
                                break
                            if out.get("yaml_text"):
                                yaml_text = str(out.get("yaml_text"))
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
            except Exception:
                pass

            if yaml_text is None:
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

            # Validate in-memory before writing
            report = validate_yaml_text(yaml_text, base_dir=output_dir or os.getcwd())
            if not report.is_valid and strict:
                typer.echo("[red]Generated YAML is invalid under --strict")
                raise typer.Exit(1)

            # Write outputs
            # Determine output location (project-aware by default)
            project_root = str(find_project_root())
            out_dir = output_dir or project_root
            os.makedirs(out_dir, exist_ok=True)
            out_yaml = os.path.join(out_dir, "pipeline.yaml")
            # In project-aware default path, allow overwriting pipeline.yaml without --force
            allow_overwrite = (output_dir is None) or (
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
            with open(out_yaml, "w") as f:
                f.write(yaml_text)
            typer.echo(f"[green]Wrote: {out_yaml}")

            # Optionally update flujo.toml budget
            try:
                if budget is not None or (not non_interactive):
                    budget_val: float
                    if budget is None and not non_interactive:
                        # typer.prompt returns a string; cast to float explicitly
                        _resp = typer.prompt(
                            "What is a safe cost limit per run (USD)?", default="2.50"
                        )
                        budget_val = float(_resp)
                    else:
                        # At this point budget is not None by guard
                        budget_val = float(budget)  # type: ignore[arg-type]
                    # Determine pipeline name to write budget under
                    pipeline_name = (
                        name or _extract_pipeline_name_from_yaml(yaml_text) or "pipeline"
                    )
                    flujo_toml_path = Path(out_dir) / "flujo.toml"
                    if flujo_toml_path.exists():
                        update_project_budget(flujo_toml_path, pipeline_name, budget_val)
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
        None, "--input", "--input-data", "-i", help="Initial input data for the pipeline"
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
            # Ensure input_data is provided or read from stdin for YAML runs
            if input_data is None:
                import sys as _sys

                if not _sys.stdin.isatty():
                    input_data = _sys.stdin.read().strip()
                else:
                    typer.echo(
                        "[red]Error: --input is required for YAML runs when no stdin is provided.",
                        err=True,
                    )
                    raise typer.Exit(1)
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

        # Create Flujo runner using helper function
        runner = create_flujo_runner(
            pipeline=pipeline_obj,
            context_model_class=context_model_class,
            initial_context_data=initial_context_data,
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

        # Handle output
        if json_output:
            typer.echo(result)
        else:
            display_pipeline_results(result, run_id, json_output)

    except Exception as e:
        try:
            import os

            os.makedirs("output", exist_ok=True)
            with open("output/last_run_error.txt", "w") as fh:
                fh.write(repr(e))
        except Exception:
            pass
        typer.echo(f"[red]Error running pipeline: {e}", err=True)
        raise typer.Exit(1)


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


# Explicit exports
__all__ = [
    "app",
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

# Register common top-level aliases expected by tests and docs
try:
    app.command(name="solve")(solve)
    app.command(name="bench")(bench)
    app.command(name="explain")(explain)
    app.command(name="improve")(improve)
    app.command(name="validate")(validate)
    app.command(name="show-config")(show_config_cmd)
    app.command(name="version-cmd")(version_cmd)
    app.command(name="pipeline-mermaid")(pipeline_mermaid_cmd)
    app.command(name="add-eval-case")(add_eval_case_cmd)
except Exception:
    pass


if __name__ == "__main__":
    try:
        app()
    except (SettingsError, ConfigurationError) as e:
        typer.echo(f"[red]Settings error: {e}[/red]", err=True)
        raise typer.Exit(2)


def get_cli_defaults(command: str) -> Dict[str, Any]:
    """Pass-through for tests to monkeypatch at flujo.cli.main level.

    Delegates to the real config manager function unless monkeypatched in tests.
    """
    return _get_cli_defaults(command)


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
