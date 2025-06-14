"""CLI entry point for pydantic-ai-orchestrator."""

import typer
import json
import os
import yaml
from pydantic_ai_orchestrator.domain.models import Task, Checklist
from pydantic_ai_orchestrator.infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    get_reflection_agent,
    make_agent_async,
    self_improvement_agent,
    REVIEW_SYS,
    SOLUTION_SYS,
    VALIDATE_SYS,
)
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.application.eval_adapter import run_pipeline_async
from pydantic_ai_orchestrator.application.self_improvement import (
    evaluate_and_improve,
    SelfImprovementAgent,
)
from pydantic_ai_orchestrator.application.pipeline_runner import PipelineRunner
from pydantic_ai_orchestrator.infra.settings import settings, SettingsError
from pydantic_ai_orchestrator.exceptions import ConfigurationError

from pydantic_ai_orchestrator.infra.telemetry import init_telemetry, logfire
from typing_extensions import Annotated
from rich.table import Table
from rich.console import Console
from pydantic_ai_orchestrator.domain import Pipeline, Step
import runpy

app = typer.Typer(rich_markup_mode="markdown")

# Initialize telemetry at the start of CLI execution
init_telemetry()


@app.command()
def solve(
    prompt: str,
    max_iters: Annotated[int, typer.Option(help="Maximum number of iterations.")] = None,
    k: Annotated[
        int, typer.Option(help="Number of solution variants to generate per iteration.")
    ] = None,
    reflection: Annotated[bool, typer.Option(help="Enable/disable reflection agent.")] = None,
    scorer: Annotated[
        str, typer.Option(help="Scoring strategy: 'ratio', 'weighted', or 'reward'.")
    ] = None,
    weights_path: Annotated[str, typer.Option(help="Path to weights file (JSON or YAML)")] = None,
    solution_model: Annotated[str, typer.Option(help="Model for the Solution agent.")] = None,
    review_model: Annotated[str, typer.Option(help="Model for the Review agent.")] = None,
    validator_model: Annotated[str, typer.Option(help="Model for the Validator agent.")] = None,
    reflection_model: Annotated[str, typer.Option(help="Model for the Reflection agent.")] = None,
):
    """
    Solves a task using the multi-agent orchestrator.

    Command-line options override environment variables and settings defaults.
    """
    try:
        # Override settings from CLI args if they are provided
        if reflection is not None:
            settings.reflection_enabled = reflection
        if scorer:
            settings.scorer = scorer

        metadata = {}
        if weights_path:
            if not os.path.isfile(weights_path):
                typer.echo(f"[red]Weights file not found: {weights_path}", err=True)
                raise typer.Exit(1)
            try:
                with open(weights_path, "r") as f:
                    if weights_path.endswith((".yaml", ".yml")):
                        weights = yaml.safe_load(f)
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

        sol_model = solution_model or settings.default_solution_model
        rev_model = review_model or settings.default_review_model
        val_model = validator_model or settings.default_validator_model
        ref_model = reflection_model or settings.default_reflection_model

        review = make_agent_async(rev_model, REVIEW_SYS, Checklist)
        solution = make_agent_async(sol_model, SOLUTION_SYS, str)
        validator = make_agent_async(val_model, VALIDATE_SYS, Checklist)
        reflection_agent = get_reflection_agent(ref_model)

        orch = Orchestrator(
            review,
            solution,
            validator,
            reflection_agent,
            max_iters=max_iters,
            k_variants=k,
            reflection_limit=settings.reflection_limit,
        )
        best = orch.run_sync(Task(prompt=prompt, metadata=metadata))
        typer.echo(json.dumps(best.model_dump(), indent=2))
    except ConfigurationError as e:
        typer.echo(f"[red]Configuration Error: {e}[/red]", err=True)
        raise typer.Exit(2)
    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)


@app.command(name="version-cmd")
def version_cmd():
    """Print the package version."""
    import importlib.metadata as importlib_metadata

    try:
        try:
            v = importlib_metadata.version("pydantic_ai_orchestrator")
        except importlib_metadata.PackageNotFoundError:
            v = "unknown"
        except Exception:
            v = "unknown"
    except Exception:
        v = "unknown"
    print(f"pydantic-ai-orchestrator version: {v}")


@app.command(name="show-config")
def show_config_cmd():
    """Print effective Settings with secrets masked."""
    typer.echo(settings.model_dump(exclude={"openai_api_key", "logfire_api_key"}))


@app.command()
def bench(prompt: str, rounds: int = 10):
    """Quick micro-benchmark of generation latency/score."""
    import time
    import numpy as np

    try:
        orch = Orchestrator(review_agent, solution_agent, validator_agent, get_reflection_agent())
        times = []
        scores = []
        for i in range(rounds):
            with logfire.span("bench_round", idx=i):
                start = time.time()
                result = orch.run_sync(Task(prompt=prompt))
                times.append(time.time() - start)
                scores.append(result.score)
                logfire.info(
                    f"Round {i + 1} completed in {times[-1]:.2f}s with score {scores[-1]:.2f}"
                )

        avg_time = sum(times) / len(times)
        avg_score = sum(scores) / len(scores)
        p50_time = float(np.percentile(times, 50))
        p95_time = float(np.percentile(times, 95))
        p50_score = float(np.percentile(scores, 50))
        p95_score = float(np.percentile(scores, 95))

        table = Table(title="Benchmark Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("p50", justify="right")
        table.add_column("p95", justify="right")
        table.add_row("Latency (s)", f"{avg_time:.2f}", f"{p50_time:.2f}", f"{p95_time:.2f}")
        table.add_row("Score", f"{avg_score:.2f}", f"{p50_score:.2f}", f"{p95_score:.2f}")
        console = Console()
        console.print(table)
    except KeyboardInterrupt:
        logfire.info("Aborted by user (KeyboardInterrupt). Closing spans and exiting.")
        raise typer.Exit(130)


@app.command()
def improve(pipeline_path: str, dataset_path: str):
    """Run evaluation and generate improvement suggestions."""
    import asyncio
    import functools

    try:
        pipe_ns = runpy.run_path(pipeline_path)
        dataset_ns = runpy.run_path(dataset_path)
    except Exception as e:  # pragma: no cover - user error
        typer.echo(f"[red]Failed to load file: {e}", err=True)
        raise typer.Exit(1)

    pipeline = pipe_ns.get("pipeline") or pipe_ns.get("PIPELINE")
    dataset = dataset_ns.get("dataset") or dataset_ns.get("DATASET")
    if not isinstance(pipeline, (Pipeline, Step)) or dataset is None:
        typer.echo("[red]Invalid pipeline or dataset file", err=True)
        raise typer.Exit(1)

    runner = PipelineRunner(pipeline)
    task_fn = functools.partial(run_pipeline_async, runner=runner)
    agent = SelfImprovementAgent(self_improvement_agent)
    report = asyncio.run(evaluate_and_improve(task_fn, dataset, agent))
    typer.echo(json.dumps(report.model_dump(), indent=2))


@app.command()
def explain(path: str):
    """Print a summary of a pipeline defined in a file."""
    try:
        ns = runpy.run_path(path)
    except Exception as e:
        typer.echo(f"[red]Failed to load pipeline file: {e}", err=True)
        raise typer.Exit(1)
    pipeline = ns.get("pipeline") or ns.get("PIPELINE")
    if not isinstance(pipeline, Pipeline):
        typer.echo("[red]No 'pipeline' variable of type Pipeline found", err=True)
        raise typer.Exit(1)
    for step in pipeline.steps:
        typer.echo(step.name)


@app.callback()
def main(
    profile: bool = typer.Option(False, "--profile", help="Enable Logfire STDOUT span viewer"),
):
    """pydantic-ai-orchestrator CLI entrypoint."""
    if profile:
        logfire.configure(send_to_logfire=False, console=True)


if __name__ == "__main__":
    try:
        app()
    except (SettingsError, ConfigurationError) as e:
        typer.echo(f"[red]Settings error: {e}", err=True)
        raise typer.Exit(2)
