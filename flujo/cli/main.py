# type: ignore
"""CLI entry point for flujo."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast
import typer
import json
import os
import yaml
from pathlib import Path
from flujo.domain.models import Task, Checklist
from flujo.infra.agents import (
    make_agent_async,
    make_self_improvement_agent,
    REVIEW_SYS,
    SOLUTION_SYS,
    VALIDATE_SYS,
    get_reflection_agent,
)
from flujo.recipes import Default
from flujo.application.eval_adapter import run_pipeline_async
from flujo.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
    ImprovementReport,
)
from flujo.domain.models import ImprovementSuggestion
from flujo.application.flujo_engine import Flujo
from flujo.infra.settings import settings
from flujo.exceptions import ConfigurationError, SettingsError
from flujo.infra.telemetry import init_telemetry, logfire
from typing_extensions import Annotated
from rich.table import Table
from rich.console import Console
from flujo.domain import Pipeline, Step
import runpy
from flujo.domain.agent_protocol import AsyncAgentProtocol

# Type definitions for CLI
WeightsType = List[Dict[str, Union[str, float]]]
MetadataType = Dict[str, Any]

ScorerChoices = ["ratio", "weighted", "reward"]

app: typer.Typer = typer.Typer(rich_markup_mode="markdown")

# Initialize telemetry at the start of CLI execution
init_telemetry()


@app.command()
def solve(
    prompt: str,
    max_iters: Annotated[Optional[int], typer.Option(help="Maximum number of iterations.")] = None,
    k: Annotated[
        Optional[int], typer.Option(help="Number of solution variants to generate per iteration.")
    ] = None,
    reflection: Annotated[
        Optional[bool], typer.Option(help="Enable/disable reflection agent.")
    ] = None,
    scorer: Annotated[
        Optional[str],
        typer.Option(
            help="Scoring strategy: 'ratio', 'weighted', or 'reward'.",
        ),
    ] = None,
    weights_path: Annotated[
        Optional[str], typer.Option(help="Path to weights file (JSON or YAML)")
    ] = None,
    solution_model: Annotated[
        Optional[str], typer.Option(help="Model for the Solution agent.")
    ] = None,
    review_model: Annotated[Optional[str], typer.Option(help="Model for the Review agent.")] = None,
    validator_model: Annotated[
        Optional[str], typer.Option(help="Model for the Validator agent.")
    ] = None,
    reflection_model: Annotated[
        Optional[str], typer.Option(help="Model for the Reflection agent.")
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
        # Argument validation
        if max_iters is not None and max_iters <= 0:
            typer.echo("[red]Error: --max-iters must be a positive integer[/red]", err=True)
            raise typer.Exit(2)
        if k is not None and k <= 0:
            typer.echo("[red]Error: --k must be a positive integer[/red]", err=True)
            raise typer.Exit(2)
        if scorer is not None and scorer not in {"ratio", "weighted", "reward"}:
            typer.echo(
                "[red]Error: --scorer must be one of 'ratio', 'weighted', or 'reward'[/red]",
                err=True,
            )
            raise typer.Exit(2)
        # Override settings from CLI args if they are provided
        if reflection is not None:
            settings.reflection_enabled = reflection
        if scorer:
            settings.scorer = scorer

        metadata: MetadataType = {}
        if weights_path:
            if not os.path.isfile(weights_path):
                typer.echo(f"[red]Weights file not found: {weights_path}", err=True)
                raise typer.Exit(1)
            try:
                with open(weights_path, "r") as f:
                    if weights_path.endswith((".yaml", ".yml")):
                        weights: WeightsType = yaml.safe_load(f)
                    else:
                        weights = json.load(f)
                if not isinstance(weights, list) or not all(
                    isinstance(w, dict) and "item" in w and "weight" in w for w in weights
                ):
                    typer.echo(
                        "[red]Weights file must be a list of objects with 'item' and 'weight'",
                        err=True,
                    )
                    raise typer.Exit(1)
                metadata["weights"] = weights
            except Exception as e:
                typer.echo(f"[red]Error loading weights file: {e}", err=True)
                raise typer.Exit(1)

        sol_model: str = solution_model or settings.default_solution_model
        rev_model: str = review_model or settings.default_review_model
        val_model: str = validator_model or settings.default_validator_model
        ref_model: str = reflection_model or settings.default_reflection_model

        review: AsyncAgentProtocol[Any, Checklist] = cast(
            AsyncAgentProtocol[Any, Checklist], make_agent_async(rev_model, REVIEW_SYS, Checklist)
        )
        solution: AsyncAgentProtocol[Any, str] = cast(
            AsyncAgentProtocol[Any, str], make_agent_async(sol_model, SOLUTION_SYS, str)
        )
        validator: AsyncAgentProtocol[Any, Checklist] = cast(
            AsyncAgentProtocol[Any, Checklist], make_agent_async(val_model, VALIDATE_SYS, Checklist)
        )
        reflection_agent: AsyncAgentProtocol[Any, str] = cast(
            AsyncAgentProtocol[Any, str], get_reflection_agent(ref_model)
        )

        orch: Default = Default(
            review,
            solution,
            validator,
            reflection_agent,
            max_iters=max_iters,
            k_variants=k,
            reflection_limit=settings.reflection_limit,
        )
        best = orch.run_sync(Task(prompt=prompt, metadata=metadata))
        if best is not None:
            typer.echo(json.dumps(best.model_dump(), indent=2))
        else:
            typer.echo("[red]No solution found[/red]", err=True)
            raise typer.Exit(1)
    except ConfigurationError as e:
        typer.echo(f"[red]Configuration Error: {e}[/red]", err=True)
        raise typer.Exit(2)
    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)


@app.command(name="version-cmd")
def version_cmd() -> None:
    """
    Print the package version.

    Returns:
        None: Prints version to stdout
    """
    import importlib.metadata as importlib_metadata

    try:
        try:
            v: str = importlib_metadata.version("flujo")
        except importlib_metadata.PackageNotFoundError:
            v = "unknown"
        except Exception:
            v = "unknown"
    except Exception:
        v = "unknown"
    print(f"flujo version: {v}")


@app.command(name="show-config")
def show_config_cmd() -> None:
    """
    Print effective Settings with secrets masked.

    Returns:
        None: Prints configuration to stdout
    """
    typer.echo(settings.model_dump(exclude={"openai_api_key", "logfire_api_key"}))


@app.command()
def bench(prompt: str, rounds: int = 10) -> None:
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
    import time
    import numpy as np

    try:
        review_agent = make_agent_async(settings.default_review_model, REVIEW_SYS, Checklist)
        solution_agent = make_agent_async(settings.default_solution_model, SOLUTION_SYS, str)
        validator_agent = make_agent_async(
            settings.default_validator_model, VALIDATE_SYS, Checklist
        )
        orch: Default = Default(
            review_agent, solution_agent, validator_agent, get_reflection_agent()
        )
        times: List[float] = []
        scores: List[float] = []
        for i in range(rounds):
            with logfire.span("bench_round", idx=i):
                start: float = time.time()
                result = orch.run_sync(Task(prompt=prompt))
                if result is not None:
                    times.append(time.time() - start)
                    scores.append(result.score)
                    logfire.info(
                        f"Round {i + 1} completed in {times[-1]:.2f}s with score {scores[-1]:.2f}"
                    )

        if not times or not scores:
            typer.echo("[red]No successful runs completed[/red]", err=True)
            raise typer.Exit(1)

        avg_time: float = sum(times) / len(times)
        avg_score: float = sum(scores) / len(scores)
        p50_time: float = float(np.percentile(times, 50))
        p95_time: float = float(np.percentile(times, 95))
        p50_score: float = float(np.percentile(scores, 50))
        p95_score: float = float(np.percentile(scores, 95))

        table: Table = Table(title="Benchmark Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("p50", justify="right")
        table.add_column("p95", justify="right")
        table.add_row("Latency (s)", f"{avg_time:.2f}", f"{p50_time:.2f}", f"{p95_time:.2f}")
        table.add_row("Score", f"{avg_score:.2f}", f"{p50_score:.2f}", f"{p95_score:.2f}")
        console: Console = Console()
        console.print(table)
    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)


@app.command(name="add-eval-case")
def add_eval_case_cmd(
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to the Python file containing the Dataset object"
    ),
    case_name: str = typer.Option(
        ..., "--name", "-n", prompt="Enter a unique name for the new evaluation case"
    ),
    inputs: str = typer.Option(
        ..., "--inputs", "-i", prompt="Enter the primary input for this case"
    ),
    expected_output: Optional[str] = typer.Option(
        None, "--expected", "-e", prompt="Enter the expected output (or skip)", show_default=False
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
            parsed = json.loads(metadata_json)
            case_parts.append(f"metadata={parsed}")
        except json.JSONDecodeError:
            typer.secho(
                f"Error: Invalid JSON provided for metadata: {metadata_json}", fg=typer.colors.RED
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
    import asyncio
    import functools

    try:
        pipe_ns: Dict[str, Any] = runpy.run_path(pipeline_path)
        dataset_ns: Dict[str, Any] = runpy.run_path(dataset_path)
    except Exception as e:  # pragma: no cover - user error handling, covered in integration tests
        typer.echo(f"[red]Failed to load file: {e}", err=True)
        raise typer.Exit(1)

    pipeline: Optional[Union[Pipeline, Step]] = pipe_ns.get("pipeline") or pipe_ns.get("PIPELINE")
    dataset: Optional[Any] = dataset_ns.get("dataset") or dataset_ns.get("DATASET")
    if not isinstance(pipeline, (Pipeline, Step)) or dataset is None:
        typer.echo("[red]Invalid pipeline or dataset file", err=True)
        raise typer.Exit(1)

    runner: Flujo = Flujo(pipeline)
    task_fn = functools.partial(run_pipeline_async, runner=runner)
    _agent = make_self_improvement_agent(model=improvement_agent_model)
    agent: SelfImprovementAgent = SelfImprovementAgent(_agent)
    report: ImprovementReport = asyncio.run(
        evaluate_and_improve(task_fn, dataset, agent, pipeline_definition=pipeline)
    )
    if json_output:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    console = Console()
    console.print("[bold]IMPROVEMENT REPORT[/bold]")
    groups: Dict[str, List[ImprovementSuggestion]] = {}
    for sugg in report.suggestions:
        key = sugg.target_step_name or "Evaluation Suite"
        groups.setdefault(key, []).append(sugg)

    for step, suggestions in groups.items():
        console.print(f"\n[bold cyan]Suggestions for {step}[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Failure Pattern")
        table.add_column("Suggestion")
        table.add_column("Impact", justify="center")
        table.add_column("Effort", justify="center")
        for s in suggestions:
            detail = s.detailed_explanation
            if s.prompt_modification_details:
                detail += f"\nPrompt: {s.prompt_modification_details.modification_instruction}"
            elif s.config_change_details:
                parts = [
                    f"{c.parameter_name}->{c.suggested_value}" for c in s.config_change_details
                ]
                detail += "\nConfig: " + ", ".join(parts)
            elif s.suggested_new_eval_case_description:
                detail += f"\nNew Case: {s.suggested_new_eval_case_description}"
            table.add_row(
                s.failure_pattern_summary,
                f"{s.suggestion_type.name}: {detail}",
                s.estimated_impact or "",
                s.estimated_effort_to_implement or "",
            )
        console.print(table)


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
        ns: Dict[str, Any] = runpy.run_path(path)
    except Exception as e:
        typer.echo(f"[red]Failed to load pipeline file: {e}", err=True)
        raise typer.Exit(1)
    pipeline: Optional[Pipeline] = ns.get("pipeline") or ns.get("PIPELINE")
    if not isinstance(pipeline, Pipeline):
        typer.echo("[red]No 'pipeline' variable of type Pipeline found", err=True)
        raise typer.Exit(1)
    for step in pipeline.steps:  # type: ignore
        typer.echo(step.name)


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
    "main",
]


if __name__ == "__main__":
    try:
        app()
    except (SettingsError, ConfigurationError) as e:
        typer.echo(f"[red]Settings error: {e}", err=True)
        raise typer.Exit(2)
