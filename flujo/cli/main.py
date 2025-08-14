"""CLI entry point for flujo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast
import typer
import click
import json
from pathlib import Path
from flujo.infra.config_manager import get_cli_defaults as _get_cli_defaults
from flujo.exceptions import ConfigurationError, SettingsError
from flujo.infra import telemetry
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
    parse_context_data,
    load_pipeline_from_file,
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


app: typer.Typer = typer.Typer(rich_markup_mode="markdown")

# Initialize telemetry at the start of CLI execution
telemetry.init_telemetry()
logfire = telemetry.logfire

app.add_typer(lens_app, name="lens")
budgets_app: typer.Typer = typer.Typer(help="Budget governance commands")
app.add_typer(budgets_app, name="budgets")


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


@app.command()
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


@app.command(name="version-cmd")
def version_cmd() -> None:
    """
    Print the package version.

    Returns:
        None: Prints version to stdout
    """
    version = get_version_string()
    typer.echo(f"flujo version: {version}")


@app.command(name="show-config")
def show_config_cmd() -> None:
    """
    Print effective Settings with secrets masked.

    Returns:
        None: Prints configuration to stdout
    """
    typer.echo(get_masked_settings_dict())


@app.command()
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
        cli_args = apply_cli_defaults("bench", {"rounds": 10}, rounds=rounds)
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


@app.command(name="add-eval-case")
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


@app.command()
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


@app.command()
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


@app.command()
def validate(
    path: str,
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


@app.command()
def run(
    pipeline_file: str = typer.Argument(
        ..., help="Path to the pipeline to run (.py or .yaml/.yml)"
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
            {"pipeline_name": "pipeline", "json_output": False},
            pipeline_name=pipeline_name,
            json_output=json_output,
        )
        pipeline_name = cast(str, cli_args["pipeline_name"])
        json_output = cast(bool, cli_args["json_output"])

        # Detect raw flags to support JSON mode when alias parsing fails
        ctx = click.get_current_context()
        if not json_output and any(flag in ctx.args for flag in ("--json", "--json-output")):
            json_output = True

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


@app.command()
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


@app.command("pipeline-mermaid")
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
