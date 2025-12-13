from __future__ import annotations
# mypy: ignore-errors

from pathlib import Path
from typing import Optional, Union

import json
import os
import typer
import click
from typing_extensions import Annotated

from flujo.exceptions import ConfigurationError
from flujo.infra import telemetry
from flujo.utils.serialization import safe_deserialize
from .helpers import (
    run_benchmark_pipeline,
    create_benchmark_table,
    setup_solve_command_environment,
    execute_solve_pipeline,
    print_rich_or_typer,
    apply_cli_defaults,
    execute_improve,
)

logfire = telemetry.logfire
ScorerType = str


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
        try:
            payload = best.model_dump(mode="json")
        except TypeError:
            payload = best.model_dump()
        typer.echo(json.dumps(payload, indent=2))

    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)
    except ConfigurationError as e:
        typer.secho(f"Configuration Error: {e}", err=True)
        raise typer.Exit(2)


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
        rounds_val = cli_args.get("rounds")
        if isinstance(rounds_val, int):
            rounds = rounds_val

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
            try:
                from flujo.cli import dev_commands as _dev

                _sd = getattr(_dev, "safe_deserialize", safe_deserialize)
            except Exception:
                _sd = safe_deserialize
            parsed = _sd(json.loads(metadata_json))
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


def register_experimental_commands(experimental_app: typer.Typer) -> None:
    experimental_app.command(name="solve")(solve)
    experimental_app.command(name="bench")(bench)
    experimental_app.command(name="add-case")(add_eval_case_cmd)
    experimental_app.command(name="improve")(improve)
