"""CLI entry point for pydantic-ai-orchestrator.""" 

import typer
import json
from pydantic_ai_orchestrator.domain.models import Task
from pydantic_ai_orchestrator.infra.agents import review_agent, solution_agent, validator_agent, reflection_agent
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.infra.settings import settings
from importlib.metadata import version
import logfire
from typing_extensions import Annotated

app = typer.Typer(rich_markup_mode="markdown")

@app.command()
def solve(
    prompt: str,
    max_iters: Annotated[int, typer.Option(help="Maximum number of iterations.")] = None,
    k: Annotated[int, typer.Option(help="Number of solution variants to generate per iteration.")] = None,
    reflection: Annotated[bool, typer.Option(help="Enable/disable reflection agent.")] = None,
    scorer: Annotated[str, typer.Option(help="Scoring strategy: 'ratio', 'weighted', or 'reward'.")] = None,
):
    """
    Solves a task using the multi-agent orchestrator.

    Command-line options override environment variables and settings defaults.
    """
    # Override settings from CLI args if they are provided
    if reflection is not None:
        settings.reflection_enabled = reflection
    if scorer:
        settings.scorer = scorer

    # The Orchestrator will use CLI args if provided, otherwise fall back to settings
    orch = Orchestrator(
        review_agent,
        solution_agent,
        validator_agent,
        reflection_agent(), # Re-initialize to respect potential setting change
        max_iters=max_iters,
        k_variants=k,
    )
    best = orch.run(Task(prompt=prompt))
    typer.echo(json.dumps(best.model_dump(), indent=2))

@app.command()
def version_cmd():
    """Show package version."""
    typer.echo(f"pydantic-ai-orchestrator version: {version('pydantic_ai_orchestrator')}")

@app.command(name="show-config")
def show_config_cmd():
    """Print effective Settings with secrets masked."""
    typer.echo(settings.model_dump(exclude={"openai_api_key", "logfire_api_key"}))

@app.command()
def bench(prompt: str, rounds: int = 10):
    """Quick micro-benchmark of generation latency/score."""
    import time
    orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent())
    times = []
    scores = []
    for i in range(rounds):
        with logfire.span("bench_round", idx=i):
            start = time.time()
            result = orch.run(Task(prompt=prompt))
            times.append(time.time() - start)
            scores.append(result.score)
            logfire.info(f"Round {i+1} completed in {times[-1]:.2f}s with score {scores[-1]:.2f}")

    avg_time = sum(times) / len(times)
    avg_score = sum(scores) / len(scores)
    typer.echo(f"\nBenchmark Complete ({rounds} rounds):")
    typer.echo(f"  Avg latency: {avg_time:.2f}s")
    typer.echo(f"  Avg score:   {avg_score:.2f}")

@app.callback()
def main(profile: bool = typer.Option(False, "--profile", help="Enable Logfire STDOUT span viewer")):
    """pydantic-ai-orchestrator CLI entrypoint."""
    if profile:
        logfire.configure(send_to_logfire=False, console=True)

if __name__ == "__main__":
    app() 