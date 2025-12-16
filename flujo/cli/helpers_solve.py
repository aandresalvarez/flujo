from __future__ import annotations

import warnings
from typing import Any, List, Optional

from typer import Exit

from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.models import Candidate, Checklist, Task
from flujo.type_definitions.common import JSONObject
from flujo.utils.async_bridge import run_sync

from .helpers_io import load_weights_file
from .helpers_project import apply_cli_defaults


def create_agents_for_solve(
    solution_model: str,
    review_model: str,
    validator_model: str,
    reflection_model: str,
) -> tuple[
    AsyncAgentProtocol[Any, str],
    AsyncAgentProtocol[Any, Checklist],
    AsyncAgentProtocol[Any, Checklist],
    AsyncAgentProtocol[Any, str],
]:
    """Create all agents needed for the solve command."""
    from flujo.cli.main import (
        make_review_agent,
        make_solution_agent,
        make_validator_agent,
        get_reflection_agent,
    )

    review = make_review_agent(review_model)
    solution = make_solution_agent(solution_model)
    validator = make_validator_agent(validator_model)
    reflection_agent = get_reflection_agent(reflection_model)

    return solution, review, validator, reflection_agent


def run_solve_pipeline(
    prompt: str,
    metadata: JSONObject,
    solution_agent: AsyncAgentProtocol[Any, str],
    review_agent: AsyncAgentProtocol[Any, Checklist],
    validator_agent: AsyncAgentProtocol[Any, Checklist],
    reflection_agent: AsyncAgentProtocol[Any, str],
    k_variants: int,
    max_iters: int,
    reflection_limit: int,
) -> Candidate | None:
    """Run the solve pipeline with the given configuration."""
    import flujo.cli.main as cli_main
    from flujo.cli.main import run_default_pipeline

    pipeline = cli_main.make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
        reflection_agent=reflection_agent,
        k_variants=k_variants,
        max_iters=max_iters,
        reflection_limit=reflection_limit,
    )

    return run_sync(run_default_pipeline(pipeline, Task(prompt=prompt, metadata=metadata)))


def run_benchmark_pipeline(
    prompt: str,
    rounds: int,
    logfire: Any,
) -> tuple[List[float], List[float]]:
    """Run benchmark pipeline for the given number of rounds."""
    from flujo.cli.main import (
        make_review_agent,
        make_solution_agent,
        make_validator_agent,
        get_reflection_agent,
    )

    review_agent = make_review_agent()
    solution_agent = make_solution_agent()
    validator_agent = make_validator_agent()

    import flujo.cli.main as cli_main

    pipeline = cli_main.make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
        reflection_agent=get_reflection_agent(),
        k_variants=1,
        max_iters=3,
    )

    times: List[float] = []
    scores: List[float] = []

    import time

    for i in range(rounds):
        with logfire.span("bench_round", idx=i):
            start = time.perf_counter()
            from flujo.cli.main import run_default_pipeline

            result: Candidate | None = run_sync(run_default_pipeline(pipeline, Task(prompt=prompt)))
            if result is not None:
                times.append(time.perf_counter() - start)
                scores.append(result.score)
                logfire.info(
                    f"Round {i + 1} completed in {times[-1]:.2f}s with score {scores[-1]:.2f}"
                )

    return times, scores


def create_benchmark_table(times: List[float], scores: List[float]) -> Any:
    """Create a rich table for benchmark results."""
    import numpy as np

    if not times or not scores:
        raise Exit(1)

    avg_time = sum(times) / len(times)
    avg_score = sum(scores) / len(scores)
    p50_time = float(np.percentile(times, 50))
    p95_time = float(np.percentile(times, 95))
    p50_score = float(np.percentile(scores, 50))
    p95_score = float(np.percentile(scores, 95))

    try:
        from rich.table import Table as _Table

        table = _Table(title="Benchmark Results", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Mean", justify="right")
        table.add_column("p50", justify="right")
        table.add_column("p95", justify="right")
        table.add_row("Latency (s)", f"{avg_time:.2f}", f"{p50_time:.2f}", f"{p95_time:.2f}")
        table.add_row("Score", f"{avg_score:.2f}", f"{p50_score:.2f}", f"{p95_score:.2f}")
        return table
    except ModuleNotFoundError:
        return (
            "Benchmark Results\n"
            "Metric | Mean | p50 | p95\n"
            f"Latency (s) | {avg_time:.2f} | {p50_time:.2f} | {p95_time:.2f}\n"
            f"Score | {avg_score:.2f} | {p50_score:.2f} | {p95_score:.2f}"
        )


def setup_json_output_mode(json_output: bool) -> None:
    """Set up JSON output mode by suppressing logging and warnings."""
    if json_output:
        import logging

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")


def create_improvement_report_table(suggestions: List[Any]) -> tuple[dict[str, List[Any]], Any]:
    """Create a table for improvement suggestions."""
    groups: dict[str, List[Any]] = {}
    for sugg in suggestions:
        key = sugg.target_step_name or "Evaluation Suite"
        groups.setdefault(key, []).append(sugg)

    try:
        from rich.table import Table as _Table

        table = _Table(show_header=True, header_style="bold magenta")
        table.add_column("Failure Pattern")
        table.add_column("Suggestion")
        table.add_column("Impact", justify="center")
        table.add_column("Effort", justify="center")
        return groups, table
    except ModuleNotFoundError:
        return groups, "Failure Pattern | Suggestion | Impact | Effort"


def format_improvement_suggestion(suggestion: Any) -> str:
    """Format an improvement suggestion for display."""
    detail = suggestion.detailed_explanation

    if suggestion.prompt_modification_details:
        detail += f"\nPrompt: {suggestion.prompt_modification_details.modification_instruction}"
    elif suggestion.config_change_details:
        parts = [
            f"{c.parameter_name}->{c.suggested_value}" for c in suggestion.config_change_details
        ]
        detail += "\nConfig: " + ", ".join(parts)
    elif suggestion.suggested_new_eval_case_description:
        detail += f"\nNew Case: {suggestion.suggested_new_eval_case_description}"

    return f"{suggestion.suggestion_type.name}: {detail}"


def create_pipeline_results_table(
    step_history: List[Any],
    *,
    show_output_column: bool = True,
    output_preview_len: Optional[int] = None,
    include_steps: Optional[List[str]] = None,
) -> Any:
    """Create a table displaying pipeline execution results."""
    use_rich = True
    try:
        from rich.table import Table as _Table

        table: Any = _Table(title="Pipeline Execution Results")
        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        if show_output_column:
            table.add_column("Output", style="white")
        table.add_column("Cost", style="yellow")
        table.add_column("Tokens", style="blue")
    except ModuleNotFoundError:
        use_rich = False
        headers = ["Step", "Status"]
        if show_output_column:
            headers.append("Output")
        headers.extend(["Cost", "Tokens"])
        table = [" | ".join(headers)]

    def add_rows(step_res: Any, prefix: str = "") -> None:
        name_attr = getattr(step_res, "step_name", None)
        if name_attr is None:
            name_attr = getattr(step_res, "name", "<unknown>")
        raw_name = name_attr
        step_name = f"{prefix}{raw_name}" if prefix else raw_name
        status = "✅" if step_res.success else "❌"
        preview_len = 100 if output_preview_len is None else max(-1, int(output_preview_len))
        out_str = str(getattr(step_res, "output", ""))
        if preview_len >= 0 and len(out_str) > preview_len:
            output = out_str[:preview_len] + "..."
        else:
            output = out_str
        cost = f"${step_res.cost_usd:.4f}" if hasattr(step_res, "cost_usd") else "N/A"
        tokens = str(step_res.token_counts) if hasattr(step_res, "token_counts") else "N/A"

        include = True
        if include_steps:
            try:
                include = str(raw_name) in set(include_steps)
            except Exception:
                include = True

        if include:
            if use_rich:
                if show_output_column:
                    table.add_row(step_name, status, output, cost, tokens)
                else:
                    table.add_row(step_name, status, cost, tokens)
            else:
                row = [str(step_name), str(status)]
                if show_output_column:
                    row.append(str(output))
                row.extend([str(cost), str(tokens)])
                table.append(" | ".join(row))

        try:
            nested_history = getattr(step_res, "step_history", None)
            if isinstance(nested_history, list) and nested_history:
                for child in nested_history:
                    add_rows(child, prefix + "  ")
        except Exception:
            pass
        try:
            out = getattr(step_res, "output", None)
            if (
                out is not None
                and hasattr(out, "step_history")
                and isinstance(getattr(out, "step_history", None), list)
                and getattr(out, "step_history")
            ):
                for child in getattr(out, "step_history"):
                    add_rows(child, prefix + "  ")
        except Exception:
            pass

    for step_res in step_history:
        add_rows(step_res)

    return table if use_rich else "\n".join(table)


def setup_solve_command_environment(
    max_iters: Optional[int],
    k: Optional[int],
    reflection: Optional[bool],
    scorer: Optional[str],
    weights_path: Optional[str],
    solution_model: Optional[str],
    review_model: Optional[str],
    validator_model: Optional[str],
    reflection_model: Optional[str],
) -> tuple[dict[str, Any], JSONObject, tuple[Any, ...]]:
    """Set up the environment for the solve command."""
    from flujo.cli.main import load_settings
    from flujo.exceptions import ConfigurationError

    try:
        settings = load_settings()
        cli_args = apply_cli_defaults(
            "solve",
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

        if cli_args["max_iters"] is not None and cli_args["max_iters"] <= 0:
            from typer import secho

            secho("Error: --max-iters must be a positive integer", err=True)
            raise Exit(2)
        if cli_args["k"] is not None and cli_args["k"] <= 0:
            from typer import secho

            secho("Error: --k must be a positive integer", err=True)
            raise Exit(2)

        if cli_args["reflection"] is not None:
            settings.reflection_enabled = cli_args["reflection"]
        if cli_args["scorer"]:
            settings.scorer = cli_args["scorer"]

        metadata: JSONObject = {}
        if cli_args["weights_path"]:
            metadata["weights"] = load_weights_file(cli_args["weights_path"])

        sol_model = cli_args["solution_model"] or getattr(
            settings, "default_solution_model", "openai:gpt-4o-mini"
        )
        rev_model = cli_args["review_model"] or getattr(
            settings, "default_review_model", "openai:gpt-4o-mini"
        )
        val_model = cli_args["validator_model"] or getattr(
            settings, "default_validator_model", "openai:gpt-4o-mini"
        )
        ref_model = cli_args["reflection_model"] or getattr(
            settings, "default_reflection_model", "openai:gpt-4o-mini"
        )

        agents = create_agents_for_solve(sol_model, rev_model, val_model, ref_model)

        return cli_args, metadata, agents

    except ConfigurationError as e:
        from typer import secho

        error_msg = getattr(e, "message", str(e))
        secho(f"Configuration Error: {error_msg}", err=True)
        raise Exit(2)


def execute_solve_pipeline(
    prompt: str,
    cli_args: dict[str, Any],
    metadata: JSONObject,
    agents: tuple[Any, ...],
    settings: Any,
) -> Any:
    """Execute the solve pipeline with the given configuration."""
    solution, review, validator, reflection_agent = agents

    best = run_solve_pipeline(
        prompt=prompt,
        metadata=metadata,
        solution_agent=solution,
        review_agent=review,
        validator_agent=validator,
        reflection_agent=reflection_agent,
        k_variants=1 if cli_args["k"] is None else cli_args["k"],
        max_iters=3 if cli_args["max_iters"] is None else cli_args["max_iters"],
        reflection_limit=settings.reflection_limit,
    )

    if best is None:
        raise Exit(1)

    return best
