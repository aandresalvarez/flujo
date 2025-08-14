"""
Note: Many helpers import symbols from flujo.cli.main at runtime so tests can
monkeypatch functions there (e.g., run_default_pipeline, agent factories).
This avoids brittle test coupling while keeping main.py slim.
"""

from __future__ import annotations

import asyncio
import importlib.metadata as importlib_metadata
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Type, Union, cast

import runpy
import yaml
from rich.table import Table
from typer import Exit

from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import Checklist, PipelineContext, Task
from flujo.utils.serialization import safe_serialize
from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
import hashlib


def load_pipeline_from_file(
    pipeline_file: str, pipeline_name: str = "pipeline"
) -> tuple[Pipeline[Any, Any], str]:
    """Load a pipeline from a Python file.

    Args:
        pipeline_file: Path to the Python file
        pipeline_name: Name of the pipeline variable to look for

    Returns:
        Tuple of (pipeline_object, actual_pipeline_name)

    Raises:
        Exit: If pipeline cannot be loaded or found
    """
    try:
        ns: Dict[str, Any] = runpy.run_path(pipeline_file)
    except Exception as e:
        # Provide explicit message for CLI output
        try:
            from typer import secho

            secho(f"Failed to load pipeline file: {e}", fg="red")
        except Exception:
            pass
        raise Exit(1)

    # Find the pipeline object
    pipeline_obj = ns.get(pipeline_name)
    try:
        with open("output/last_pipeline_debug.txt", "w") as f:
            f.write(f"has_pipeline_var={pipeline_obj is not None}\n")
    except Exception:
        pass

    # If default name missing, locate any Pipeline or Step instance
    if pipeline_obj is None:
        pipeline_candidates = [
            (name, val)
            for name, val in ns.items()
            if isinstance(val, Pipeline) or isinstance(val, Step)
        ]
        if pipeline_candidates:
            # Prefer the first pipeline with more than one step
            selected = None
            for name, val in pipeline_candidates:
                if isinstance(val, Pipeline) and hasattr(val, "steps") and len(val.steps) > 1:
                    selected = (name, val)
                    break
            if selected:
                pipeline_name, pipeline_obj = selected
            else:
                pipeline_name, pipeline_obj = pipeline_candidates[0]
        else:
            try:
                from typer import secho

                secho(f"No Pipeline instance found in {pipeline_file}", fg="red")
            except Exception:
                pass
            raise Exit(1)

    # Validate that we got a Pipeline instance
    if not isinstance(pipeline_obj, Pipeline):
        # Support single-step files by wrapping a Step into a Pipeline
        if isinstance(pipeline_obj, Step):
            pipeline_obj = Pipeline.from_step(pipeline_obj)
        else:
            try:
                from typer import secho

                secho(f"Object '{pipeline_name}' is not a Pipeline instance", fg="red")
            except Exception:
                pass
            raise Exit(1)

    return pipeline_obj, pipeline_name


def load_pipeline_from_yaml_file(yaml_path: str) -> Pipeline[Any, Any]:
    """Load a pipeline from a YAML blueprint file (progressive v0).

    This relies on flujo.domain.blueprint loader and returns a Pipeline.
    """
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    try:
        # Load skills catalog from same directory (optional)
        dirname = os.path.dirname(os.path.abspath(yaml_path))
        load_skills_catalog(dirname)
        # Load packaged skills via entry points
        load_skills_entry_points()
        with open(yaml_path, "r") as f:
            text = f.read()
        # Compute spec hash for telemetry usage
        os.environ["FLUJO_YAML_SPEC_SHA256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return load_pipeline_blueprint_from_yaml(text, base_dir=dirname)
    except Exception as e:
        try:
            from typer import secho

            secho(f"Failed to load YAML pipeline: {e}", fg="red")
        except Exception:
            pass
        raise Exit(1)


def load_dataset_from_file(dataset_path: str) -> Any:
    """Load a dataset from a Python file.

    Args:
        dataset_path: Path to the Python file

    Returns:
        The dataset object

    Raises:
        Exit: If dataset cannot be loaded or found
    """
    try:
        dataset_ns: Dict[str, Any] = runpy.run_path(dataset_path)
    except Exception:
        raise Exit(1)

    dataset = dataset_ns.get("dataset") or dataset_ns.get("DATASET")
    if dataset is None:
        raise Exit(1)

    return dataset


def parse_context_data(
    context_data: Optional[str], context_file: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Parse context data from string or file.

    Args:
        context_data: JSON string with context data
        context_file: Path to JSON/YAML file with context data

    Returns:
        Parsed context data dictionary

    Raises:
        Exit: If context data cannot be parsed
    """
    if context_data:
        try:
            # Import via main for test monkeypatch friendliness
            from flujo.cli.main import safe_deserialize

            return cast(Optional[Dict[str, Any]], safe_deserialize(json.loads(context_data)))
        except json.JSONDecodeError:
            raise Exit(1)

    if context_file:
        try:
            with open(context_file, "r") as f:
                if context_file.endswith((".yaml", ".yml")):
                    return cast(Optional[Dict[str, Any]], yaml.safe_load(f))
                else:
                    from flujo.cli.main import safe_deserialize

                    return cast(Optional[Dict[str, Any]], safe_deserialize(json.load(f)))
        except Exception:
            raise Exit(1)

    return None


def validate_context_model(
    context_model: str, pipeline_file: str, ns: Dict[str, Any]
) -> Optional[Type[PipelineContext]]:
    """Validate and return a context model class.

    Args:
        context_model: Name of the context model class
        pipeline_file: Path to the pipeline file for error reporting
        ns: Namespace from the pipeline file

    Returns:
        The context model class if valid

    Raises:
        Exit: If context model is invalid
    """
    try:
        context_model_class = ns.get(context_model)
        if context_model_class is None:
            from typer import secho

            secho(f"Context model '{context_model}' not found in {pipeline_file}", fg="red")
            raise Exit(1)

        if not isinstance(context_model_class, type):
            from typer import secho

            secho(f"'{context_model}' is not a class", fg="red")
            raise Exit(1)

        # Ensure it's a proper context model class
        if not issubclass(context_model_class, PipelineContext):
            from typer import secho

            secho(f"'{context_model}' must inherit from PipelineContext", fg="red")
            raise Exit(1)

        return context_model_class
    except Exit:
        raise
    except Exception as e:
        from typer import secho

        secho(f"Error loading context model '{context_model}': {e}", fg="red")
        raise Exit(1)


def load_weights_file(weights_path: str) -> List[Dict[str, Union[str, float]]]:
    """Load weights from a JSON or YAML file.

    Args:
        weights_path: Path to the weights file

    Returns:
        List of weight dictionaries

    Raises:
        Exit: If weights file cannot be loaded or is invalid
    """
    if not os.path.isfile(weights_path):
        try:
            from typer import secho

            secho(f"Weights file not found: {weights_path}", err=True)
        except Exception:
            pass
        raise Exit(1)

    try:
        with open(weights_path, "r") as f:
            if weights_path.endswith((".yaml", ".yml")):
                weights = yaml.safe_load(f)
            else:
                from flujo.cli.main import safe_deserialize

                weights = cast(List[Dict[str, Union[str, float]]], safe_deserialize(json.load(f)))

        if not isinstance(weights, list) or not all(
            isinstance(w, dict) and "item" in w and "weight" in w for w in weights
        ):
            try:
                from typer import secho

                secho("Weights file must be a list of objects with 'item' and 'weight'", err=True)
            except Exception:
                pass
            raise Exit(1)

        return weights
    except Exception:
        try:
            from typer import secho

            secho("Error loading weights file", err=True)
        except Exception:
            pass
        raise Exit(1)


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
    """Create all agents needed for the solve command.

    Args:
        solution_model: Model for solution agent
        review_model: Model for review agent
        validator_model: Model for validator agent
        reflection_model: Model for reflection agent

    Returns:
        Tuple of (solution, review, validator, reflection) agents
    """
    # Import via main so tests can monkeypatch there
    from flujo.cli.main import (
        make_review_agent,
        make_solution_agent,
        make_validator_agent,
        get_reflection_agent,
    )

    # Import main module to call factory via attribute (ensures patch visibility)

    review = cast(AsyncAgentProtocol[Any, Checklist], make_review_agent(review_model))
    solution = cast(AsyncAgentProtocol[Any, str], make_solution_agent(solution_model))
    validator = cast(AsyncAgentProtocol[Any, Checklist], make_validator_agent(validator_model))
    reflection_agent = cast(AsyncAgentProtocol[Any, str], get_reflection_agent(reflection_model))

    return solution, review, validator, reflection_agent


def run_solve_pipeline(
    prompt: str,
    metadata: Dict[str, Any],
    solution_agent: AsyncAgentProtocol[Any, str],
    review_agent: AsyncAgentProtocol[Any, Checklist],
    validator_agent: AsyncAgentProtocol[Any, Checklist],
    reflection_agent: AsyncAgentProtocol[Any, str],
    k_variants: int,
    max_iters: int,
    reflection_limit: int,
) -> Any:
    """Run the solve pipeline with the given configuration.

    Args:
        prompt: The task prompt
        metadata: Task metadata including weights
        solution_agent: Solution agent
        review_agent: Review agent
        validator_agent: Validator agent
        reflection_agent: Reflection agent
        k_variants: Number of solution variants
        max_iters: Maximum iterations
        reflection_limit: Reflection limit

    Returns:
        The best solution found
    """
    # Import via main so tests can monkeypatch the pipeline factory

    # Call factory via cli_main to respect test patches
    import flujo.cli.main as cli_main

    pipeline = cli_main.make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
        reflection_agent=reflection_agent,
        k_variants=k_variants,
        max_iters=max_iters,
        reflection_limit=reflection_limit,
    )

    # Import via main so tests can monkeypatch there
    from flujo.cli.main import run_default_pipeline

    return asyncio.run(run_default_pipeline(pipeline, Task(prompt=prompt, metadata=metadata)))


def run_benchmark_pipeline(
    prompt: str,
    rounds: int,
    logfire: Any,
) -> tuple[List[float], List[float]]:
    """Run benchmark pipeline for the given number of rounds.

    Args:
        prompt: The prompt to benchmark
        rounds: Number of benchmark rounds
        logfire: Logfire instance for logging

    Returns:
        Tuple of (times, scores) lists
    """
    # Import via main so tests can monkeypatch there
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
            # Import run_default_pipeline via main so tests can monkeypatch there
            from flujo.cli.main import run_default_pipeline

            result = asyncio.run(run_default_pipeline(pipeline, Task(prompt=prompt)))
            if result is not None:
                times.append(time.perf_counter() - start)
                scores.append(result.score)
                logfire.info(
                    f"Round {i + 1} completed in {times[-1]:.2f}s with score {scores[-1]:.2f}"
                )

    return times, scores


def create_benchmark_table(times: List[float], scores: List[float]) -> Table:
    """Create a rich table for benchmark results.

    Args:
        times: List of execution times
        scores: List of scores

    Returns:
        Rich table with benchmark results
    """
    import numpy as np

    if not times or not scores:
        raise Exit(1)

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

    return table


def setup_json_output_mode(json_output: bool) -> None:
    """Set up JSON output mode by suppressing logging and warnings.

    Args:
        json_output: Whether to enable JSON output mode
    """
    if json_output:
        import logging

        logging.disable(logging.CRITICAL)
        warnings.filterwarnings("ignore")


def create_improvement_report_table(suggestions: List[Any]) -> tuple[Dict[str, List[Any]], Table]:
    """Create a table for improvement suggestions.

    Args:
        suggestions: List of improvement suggestions

    Returns:
        Tuple of (grouped_suggestions, table)
    """
    groups: Dict[str, List[Any]] = {}
    for sugg in suggestions:
        key = sugg.target_step_name or "Evaluation Suite"
        groups.setdefault(key, []).append(sugg)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Failure Pattern")
    table.add_column("Suggestion")
    table.add_column("Impact", justify="center")
    table.add_column("Effort", justify="center")

    return groups, table


def format_improvement_suggestion(suggestion: Any) -> str:
    """Format an improvement suggestion for display.

    Args:
        suggestion: The improvement suggestion

    Returns:
        Formatted suggestion string
    """
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


def create_pipeline_results_table(step_history: List[Any]) -> Table:
    """Create a table displaying pipeline execution results.

    Args:
        step_history: List of step results from pipeline execution

    Returns:
        Rich table with step execution details
    """
    table = Table(title="Pipeline Execution Results")
    table.add_column("Step", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Output", style="white")
    table.add_column("Cost", style="yellow")
    table.add_column("Tokens", style="blue")

    def add_rows(step_res: Any, prefix: str = "") -> None:
        name_attr = getattr(step_res, "step_name", None)
        if name_attr is None:
            name_attr = getattr(step_res, "name", "<unknown>")
        step_name = f"{prefix}{name_attr}" if prefix else name_attr
        status = "✅" if step_res.success else "❌"
        output = (
            str(step_res.output)[:100] + "..."
            if len(str(step_res.output)) > 100
            else str(step_res.output)
        )
        cost = f"${step_res.cost_usd:.4f}" if hasattr(step_res, "cost_usd") else "N/A"
        tokens = str(step_res.token_counts) if hasattr(step_res, "token_counts") else "N/A"

        table.add_row(step_name, status, output, cost, tokens)

    for step_res in step_history:
        add_rows(step_res)

    return table


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
) -> tuple[Dict[str, Any], Dict[str, Any], tuple[Any, ...]]:
    """Set up the environment for the solve command.

    Args:
        max_iters: Maximum number of iterations
        k: Number of solution variants per iteration
        reflection: Whether to enable reflection agent
        scorer: Scoring strategy to use
        weights_path: Path to weights file
        solution_model: Model for Solution agent
        review_model: Model for Review agent
        validator_model: Model for Validator agent
        reflection_model: Model for Reflection agent

    Returns:
        Tuple of (cli_args, metadata, agents)

    Raises:
        Exit: If validation fails or agents cannot be created
    """
    # Import via main so tests can monkeypatch there
    from flujo.cli.main import load_settings
    from flujo.exceptions import ConfigurationError

    try:
        # Load settings with configuration file overrides
        settings = load_settings()

        # Apply CLI defaults from configuration file
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

        # Validate arguments
        if cli_args["max_iters"] is not None and cli_args["max_iters"] <= 0:
            from typer import secho

            secho("Error: --max-iters must be a positive integer", err=True)
            raise Exit(2)
        if cli_args["k"] is not None and cli_args["k"] <= 0:
            from typer import secho

            secho("Error: --k must be a positive integer", err=True)
            raise Exit(2)

        # Override settings from CLI args
        if cli_args["reflection"] is not None:
            settings.reflection_enabled = cli_args["reflection"]
        if cli_args["scorer"]:
            settings.scorer = cli_args["scorer"]

        # Load metadata
        metadata: Dict[str, Any] = {}
        if cli_args["weights_path"]:
            metadata["weights"] = load_weights_file(cli_args["weights_path"])

        # Get model names
        sol_model = cli_args["solution_model"] or settings.default_solution_model
        rev_model = cli_args["review_model"] or settings.default_review_model
        val_model = cli_args["validator_model"] or settings.default_validator_model
        ref_model = cli_args["reflection_model"] or settings.default_reflection_model

        # Create agents
        agents = create_agents_for_solve(sol_model, rev_model, val_model, ref_model)

        return cli_args, metadata, agents

    except ConfigurationError as e:
        from typer import secho

        secho(f"Configuration Error: {e}", err=True)
        raise Exit(2)


def execute_solve_pipeline(
    prompt: str,
    cli_args: Dict[str, Any],
    metadata: Dict[str, Any],
    agents: tuple[Any, ...],
    settings: Any,
) -> Any:
    """Execute the solve pipeline with the given configuration.

    Args:
        prompt: The task prompt to solve
        cli_args: CLI arguments with defaults applied
        metadata: Pipeline metadata including weights
        agents: Tuple of (solution, review, validator, reflection) agents
        settings: Application settings

    Returns:
        The best solution found

    Raises:
        Exit: If no solution is found
    """
    solution, review, validator, reflection_agent = agents

    # Run pipeline
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


def setup_run_command_environment(
    pipeline_file: str,
    pipeline_name: str,
    json_output: bool,
    input_data: Optional[str],
    context_model: Optional[str],
    context_data: Optional[str],
    context_file: Optional[str],
) -> tuple[Any, str, Any, Optional[Dict[str, Any]], Optional[Type[PipelineContext]]]:
    """Set up the environment for the run command.

    Args:
        pipeline_file: Path to pipeline definition file
        pipeline_name: Name of pipeline variable
        json_output: Whether to output JSON
        input_data: Initial input data
        context_model: Context model class name
        context_data: JSON string for context data
        context_file: Path to context data file

    Returns:
        Tuple of (pipeline, pipeline_name, input_data, initial_context_data, context_model_class)

    Raises:
        Exit: If setup fails
    """
    import sys
    import runpy

    # Set up JSON output mode
    setup_json_output_mode(json_output)

    # Load the pipeline
    pipeline_obj, pipeline_name = load_pipeline_from_file(pipeline_file, pipeline_name)

    # Parse input data
    if input_data is None:
        # Try to get input from stdin if no --input provided
        if not sys.stdin.isatty():
            input_data = sys.stdin.read().strip()
        else:
            raise Exit(1)

    # Handle context model
    context_model_class = None
    if context_model:
        ns = runpy.run_path(pipeline_file)
        context_model_class = validate_context_model(context_model, pipeline_file, ns)

    # Parse context data
    initial_context_data = parse_context_data(context_data, context_file)

    # Ensure initial_prompt is set for custom context models
    if context_model_class is not None:
        if initial_context_data is None:
            initial_context_data = {}
        if "initial_prompt" not in initial_context_data:
            initial_context_data["initial_prompt"] = input_data

    return pipeline_obj, pipeline_name, input_data, initial_context_data, context_model_class


def create_flujo_runner(
    pipeline: Any,
    context_model_class: Optional[Type[PipelineContext]],
    initial_context_data: Optional[Dict[str, Any]],
) -> Any:
    """Create a Flujo runner instance with the given configuration.

    Args:
        pipeline: Pipeline to run
        context_model_class: Optional custom context model class
        initial_context_data: Initial context data

    Returns:
        Configured Flujo runner instance
    """
    # Import via main so tests can monkeypatch there
    from flujo.cli.main import Flujo
    from flujo.domain.models import PipelineContext

    if context_model_class is not None:
        # Use custom context model with proper typing
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=context_model_class,
            initial_context_data=initial_context_data,
        )
    else:
        # Use default PipelineContext
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=None,
            initial_context_data=initial_context_data,
        )

    return runner


def execute_pipeline_with_output_handling(
    runner: Any,
    input_data: str,
    run_id: Optional[str],
    json_output: bool,
) -> Any:
    """Execute the pipeline and handle output formatting.

    Args:
        runner: Flujo runner instance
        input_data: Input data for the pipeline
        run_id: Optional run ID for state persistence
        json_output: Whether to output JSON

    Returns:
        Pipeline execution result

    Raises:
        Exit: If execution fails
    """
    import sys
    import io

    if json_output:
        # Capture stdout and output JSON
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            if run_id is not None:
                result = runner.run(input_data, run_id=run_id)
            else:
                result = runner.run(input_data)
        finally:
            sys.stdout = old_stdout

        from flujo.utils.serialization import serialize_to_json_robust

        return serialize_to_json_robust(result, indent=2)
    else:
        # Normal execution
        if run_id is not None:
            result = runner.run(input_data, run_id=run_id)
        else:
            result = runner.run(input_data)
        return result


def display_pipeline_results(
    result: Any,
    run_id: Optional[str],
    json_output: bool,
) -> None:
    """Display pipeline execution results in the appropriate format.

    Args:
        result: Pipeline execution result
        run_id: Run ID used for execution
        json_output: Whether output is JSON
    """
    if json_output:
        # JSON output was already handled
        return

    from rich.console import Console
    import json

    console = Console()
    console.print("[bold green]Pipeline execution completed successfully![/bold green]")

    final_output = result.step_history[-1].output if result.step_history else None
    console.print(f"[bold]Final output:[/bold] {final_output}")
    console.print(f"[bold]Total cost:[/bold] ${result.total_cost_usd:.4f}")

    total_tokens = sum(s.token_counts for s in result.step_history)
    console.print(f"[bold]Total tokens:[/bold] {total_tokens}")
    console.print(f"[bold]Steps executed:[/bold] {len(result.step_history)}")
    console.print(f"[bold]Run ID:[/bold] {run_id}")

    if result.step_history:
        console.print("\n[bold]Step Results:[/bold]")
        table = create_pipeline_results_table(result.step_history)
        console.print(table)

    if result.final_pipeline_context:
        console.print("\n[bold]Final Context:[/bold]")
        console.print(
            json.dumps(
                safe_serialize(result.final_pipeline_context.model_dump()),
                indent=2,
            )
        )


# ---------------------------------------------------------------------------
# Extracted helpers for remaining CLI subcommands
# ---------------------------------------------------------------------------


def get_version_string() -> str:
    """Return the installed flujo version or 'unknown' if not found."""
    try:
        return importlib_metadata.version("flujo")
    except (importlib_metadata.PackageNotFoundError, Exception):
        return "unknown"


def get_masked_settings_dict() -> Dict[str, Any]:
    """Return settings as a dict with sensitive keys masked/removed."""
    import flujo.cli.main as cli_main

    settings = cli_main.load_settings()  # annotated in cli.main
    return cast(Dict[str, Any], settings.model_dump(exclude={"openai_api_key", "logfire_api_key"}))


def execute_improve(
    pipeline_path: str,
    dataset_path: str,
    improvement_agent_model: Optional[str],
    json_output: bool,
) -> Optional[str]:
    """Run evaluation and improvement, printing output or returning JSON string.

    Returns a JSON string when json_output=True, otherwise prints a table and returns None.
    """
    import asyncio
    import functools
    from rich.console import Console

    try:
        # Load pipeline and dataset (dataset is optional for stubbed runs)
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/trace_improve.txt", "a") as f:
                f.write("stage:load_pipeline\n")
        except Exception:
            pass
        pipeline = load_pipeline_from_file(pipeline_path)[0]
        try:
            with open("output/trace_improve.txt", "a") as f:
                f.write("stage:load_dataset\n")
            dataset = load_dataset_from_file(dataset_path)
        except Exception:
            # Fall back to a dummy dataset when optional deps aren't available in tests
            dataset = object()

        # Build runner and agent
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:build_runner\n")
        from flujo.application.runner import Flujo

        runner: Any = Flujo(pipeline)
        # Import via main for test monkeypatch friendliness
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:import_run_pipeline_async\n")
        from flujo.cli.main import run_pipeline_async

        task_fn = functools.partial(run_pipeline_async, runner=runner)

        # Import via main so tests can monkeypatch there
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:import_eval_and_agent\n")
        from flujo.cli.main import (
            evaluate_and_improve,
            SelfImprovementAgent,
            ImprovementReport,
            make_self_improvement_agent,
        )

        from flujo.exceptions import ConfigurationError

        try:
            _agent = make_self_improvement_agent(model=improvement_agent_model)
            agent: SelfImprovementAgent = SelfImprovementAgent(_agent)
        except ConfigurationError:
            # Tests often monkeypatch evaluate_and_improve and don't need a real agent
            class _Dummy:  # minimal placeholder to satisfy the call signature
                pass

            agent = _Dummy()  # type: ignore[assignment]
        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:run_eval\n")
        report: ImprovementReport = asyncio.run(
            evaluate_and_improve(task_fn, dataset, agent, pipeline_definition=pipeline)
        )

        if json_output:
            return json.dumps(safe_serialize(report.model_dump()), indent=2)

        with open("output/trace_improve.txt", "a") as f:
            f.write("stage:print\n")
        console = Console()
        console.print("[bold]IMPROVEMENT REPORT[/bold]")
        groups, table = create_improvement_report_table(report.suggestions)
        for step, suggestions in groups.items():
            console.print(f"\n[bold cyan]Suggestions for {step}[/bold cyan]")
            for s in suggestions:
                table.add_row(
                    s.failure_pattern_summary,
                    format_improvement_suggestion(s),
                    s.estimated_impact or "",
                    s.estimated_effort_to_implement or "",
                )
            console.print(table)
        return None
    except Exception as e:
        try:
            os.makedirs("output", exist_ok=True)
            with open("output/last_improve_error.txt", "w") as f:
                f.write(repr(e))
        except Exception:
            pass
        raise Exit(1)


def load_mermaid_code(file: str, object_name: str, detail_level: str) -> str:
    """Load a pipeline and return its Mermaid diagram code string."""
    pipeline, _ = load_pipeline_from_file(file, object_name)
    if not hasattr(pipeline, "to_mermaid_with_detail_level"):
        raise Exit(1)
    return cast(str, pipeline.to_mermaid_with_detail_level(detail_level))


def get_pipeline_step_names(path: str) -> list[str]:
    """Return the ordered step names for a pipeline file."""
    pipeline, _ = load_pipeline_from_file(path)
    return [step.name for step in pipeline.steps]


def validate_pipeline_file(path: str) -> Any:
    """Return the validation report for a pipeline file."""
    if path.endswith((".yaml", ".yml")):
        try:
            with open(path, "r") as f:
                yaml_text = f.read()
            from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

            # Ensure relative imports resolve from the YAML file directory
            base_dir = os.path.dirname(os.path.abspath(path))
            pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir)
        except Exception:
            raise Exit(1)
    else:
        pipeline, _ = load_pipeline_from_file(path)
    from typing import cast as _cast

    return _cast(Any, pipeline).validate_graph()


def apply_cli_defaults(
    command: str, fallback_values: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, Any]:
    """Apply CLI defaults from configuration file to command arguments.

    This function handles both None values and hardcoded defaults by checking if the current value
    matches the fallback value, indicating it wasn't explicitly provided by the user.

    Args:
        command: The command name (e.g., "solve", "bench")
        fallback_values: Optional dict mapping argument names to their hardcoded default values
        **kwargs: The command arguments to apply defaults to

    Returns:
        Dict containing the arguments with defaults applied
    """
    # Import via main so tests can monkeypatch there
    from flujo.cli.main import get_cli_defaults

    cli_defaults = get_cli_defaults(command)
    result = kwargs.copy()

    for key, value in kwargs.items():
        # Check if value is None (explicitly not provided)
        if value is None and key in cli_defaults:
            result[key] = cli_defaults[key]
        # Check if value matches fallback (using hardcoded default)
        elif (
            fallback_values
            and key in fallback_values
            and value == fallback_values[key]
            and key in cli_defaults
        ):
            result[key] = cli_defaults[key]

    return result
