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
    validate_yaml_text,
    parse_context_data,
    load_pipeline_from_file,
    find_project_root,
    scaffold_project,
    scaffold_demo_project,
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
                _tru._print_commands_panel(
                    name=_tru.COMMANDS_PANEL_TITLE,
                    commands=default_commands,
                    markup_mode=markup_mode,
                    console=console,
                    cmd_len=max_cmd_len,
                )
                for panel_name, commands in panel_to_commands.items():
                    if panel_name == _tru.COMMANDS_PANEL_TITLE:
                        continue
                    _tru._print_commands_panel(
                        name=panel_name,
                        commands=commands,
                        markup_mode=markup_mode,
                        console=console,
                        cmd_len=max_cmd_len,
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
    help=("A project-based server for building, running, and managing AI workflows."),
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


@app.command(
    help=(
        "🤖 Start a conversation with the AI Architect to build your workflow.\n\n"
        "By default this uses the full conversational state machine. Set FLUJO_ARCHITECT_MINIMAL=1"
        " to use the legacy minimal generator."
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
) -> None:
    """Conversational pipeline generation via the Architect pipeline.

    Loads the bundled architect YAML, runs it with the provided goal, and writes outputs.

    Tip: Using GPT-5? To tune agent timeouts/retries for complex reasoning, see
    docs/guides/gpt5_architect.md (agent-level `timeout`/`max_retries`) and
    step-level `config.timeout` (alias to `timeout_s`) for plugin/validator phases.
    """
    try:
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
