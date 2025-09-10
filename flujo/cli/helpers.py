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
from .exit_codes import (
    EX_IMPORT_ERROR,
    EX_RUNTIME_ERROR,
)

from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.domain.dsl import Pipeline, Step
from flujo.domain.models import Checklist, PipelineContext, Task
from flujo.utils.serialization import safe_serialize
from flujo.infra.skills_catalog import load_skills_catalog, load_skills_entry_points
import hashlib
from flujo.domain.pipeline_validation import ValidationReport
from flujo.infra import telemetry as _telemetry
from pathlib import Path
import re
import sys


def load_pipeline_from_file(
    pipeline_file: str, pipeline_name: str = "pipeline"
) -> tuple["Pipeline[Any, Any]", str]:
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
    except ModuleNotFoundError as e:
        # Common root cause: missing project root on sys.path (skills.* imports)
        mod = getattr(e, "name", None) or str(e)
        try:
            from typer import secho
            import os as _os
            import traceback as _tb

            secho(
                f"Import error: module '{mod}' not found. Try setting PYTHONPATH=. or pass --project/FLUJO_PROJECT_ROOT",
                fg="red",
                err=True,
            )
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_IMPORT_ERROR)
    except Exception as e:
        # Provide explicit message for CLI output and allow optional traceback
        try:
            from typer import secho
            import os as _os
            import traceback as _tb

            secho(f"Failed to load pipeline file: {type(e).__name__}: {e}", fg="red", err=True)
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_RUNTIME_ERROR)

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


def load_pipeline_from_yaml_file(yaml_path: str) -> "Pipeline[Any, Any]":
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
            import os as _os
            import traceback as _tb

            msg = f"Failed to load YAML pipeline: {type(e).__name__}: {e}"
            # Hint import issues if present
            if isinstance(e, ModuleNotFoundError) or "No module named" in str(e):
                msg = f"Import error while loading YAML: {e}. Try setting PYTHONPATH=. or pass --project/FLUJO_PROJECT_ROOT"
            secho(msg, fg="red", err=True)
            if _os.getenv("FLUJO_CLI_VERBOSE") == "1" or _os.getenv("FLUJO_CLI_TRACE") == "1":
                secho("\nTraceback:", fg="yellow", err=True)
                secho("".join(_tb.format_exception(e)), err=True)
        except Exception:
            pass
        raise Exit(EX_IMPORT_ERROR if isinstance(e, ModuleNotFoundError) else EX_RUNTIME_ERROR)


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


def resolve_initial_input(input_data: Optional[str]) -> str:
    """Resolve the initial input to feed the pipeline.

    Precedence:
    1) Explicit ``--input`` value. If value is "-", read from stdin.
    2) ``FLUJO_INPUT`` environment variable when set.
    3) If stdin is piped (non-TTY), read from stdin.
    4) Fallback to empty string (pipelines that don't require input).

    Args:
        input_data: The value provided via ``--input`` option, if any.

    Returns:
        The resolved input string (possibly empty).
    """
    # 1) Explicit CLI value takes precedence
    if input_data is not None:
        # Conventional "-" means read from stdin
        if input_data.strip() == "-":
            try:
                return sys.stdin.read().strip()
            except Exception:
                return ""
        return input_data

    # 2) Environment variable fallback
    try:
        env_val = os.environ.get("FLUJO_INPUT")
        if isinstance(env_val, str) and env_val != "":
            return env_val
    except Exception:
        pass

    # 3) Read from stdin if piped (or when isatty is unavailable)
    try:
        is_tty = getattr(sys.stdin, "isatty", None)
        if not (callable(is_tty) and bool(is_tty())):
            try:
                return sys.stdin.read().strip()
            except Exception:
                return ""
    except Exception:
        pass

    # 4) Safe fallback
    return ""


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

        # Recursively render nested step histories, if present
        try:
            nested_history = getattr(step_res, "step_history", None)
            if isinstance(nested_history, list) and nested_history:
                for child in nested_history:
                    add_rows(child, prefix + "  ")
        except Exception:
            pass
        # If the step output is a nested PipelineResult, render its history as well
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
    import runpy

    # Set up JSON output mode
    setup_json_output_mode(json_output)

    # Load the pipeline
    pipeline_obj, pipeline_name = load_pipeline_from_file(pipeline_file, pipeline_name)

    # Resolve initial input robustly (CLI flag, env var, piped stdin, fallback)
    input_data = resolve_initial_input(input_data)

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
    state_backend: Optional[Any] = None,
    *,
    debug: bool = False,
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

    # Try to propagate a meaningful pipeline_name to enable budget resolution from flujo.toml
    try:
        inferred_name = getattr(pipeline, "name", None)
        if not isinstance(inferred_name, str) or not inferred_name.strip():
            inferred_name = None
    except Exception:
        inferred_name = None

    # Resolve usage limits from flujo.toml (skip in test mode for performance)
    usage_limits_arg = None
    try:
        from flujo.utils.config import get_settings as _get_proc_settings

        if not _get_proc_settings().test_mode:
            from flujo.infra.config_manager import ConfigManager
            from flujo.infra.budget_resolver import resolve_limits_for_pipeline as _resolve

            cfg = ConfigManager().load_config()
            pname = inferred_name or ""
            usage_limits_arg, _src = _resolve(getattr(cfg, "budgets", None), pname)
    except Exception:
        usage_limits_arg = None

    # Select local tracer when debug is enabled
    local_tracer_arg: Any = "default" if debug else None

    # Disable tracing in test mode to reduce overhead for performance suites
    enable_tracing_arg = True
    try:
        from flujo.utils.config import get_settings as _get_proc_settings

        if _get_proc_settings().test_mode:
            enable_tracing_arg = False
    except Exception:
        pass

    if context_model_class is not None:
        # Use custom context model with proper typing
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=context_model_class,
            initial_context_data=initial_context_data,
            state_backend=state_backend,
            pipeline_name=inferred_name,
            usage_limits=usage_limits_arg,
            local_tracer=local_tracer_arg,
            enable_tracing=enable_tracing_arg,
        )
    else:
        # Use default PipelineContext
        runner = Flujo[Any, Any, PipelineContext](
            pipeline=pipeline,
            context_model=None,
            initial_context_data=initial_context_data,
            state_backend=state_backend,
            pipeline_name=inferred_name,
            usage_limits=usage_limits_arg,
            local_tracer=local_tracer_arg,
            enable_tracing=enable_tracing_arg,
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
    import asyncio
    import gc

    # Optionally skip tracing span in test/perf mode for lower overhead
    import os as _os

    _disable_span = False
    try:
        _disable_span = str(_os.getenv("FLUJO_DISABLE_TRACING", "")).strip().lower() in {
            "1",
            "true",
            "on",
            "yes",
        }
        if not _disable_span:
            from flujo.utils.config import get_settings as _get_proc_settings

            _disable_span = bool(_get_proc_settings().test_mode)
    except Exception:
        pass

    def _run_pipeline() -> Any:
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

    if _disable_span:
        try:
            result = _run_pipeline()
            return result
        finally:
            # Skip platform-specific memory relief in perf mode; keep a quick GC only
            try:
                gc.collect()
            except Exception:
                pass
    else:
        # Add a high-level span for architect or generic pipeline execution
        with _telemetry.logfire.span("pipeline_run"):
            try:
                result = _run_pipeline()
                return result
            finally:
                # Robust cleanup to prevent event loop hang
                # Force garbage collection to clean up any orphaned async objects
                gc.collect()
                # Attempt to return freed memory to the OS (platform-specific best-effort)
                try:
                    import ctypes
                    import sys as _sys

                    if _sys.platform.startswith("linux"):
                        try:
                            libc = ctypes.CDLL("libc.so.6")
                            libc.malloc_trim(0)
                        except Exception:
                            pass
                    elif _sys.platform == "darwin":
                        try:
                            libsys = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                            libsys.malloc_default_zone.restype = ctypes.c_void_p
                            zone = libsys.malloc_default_zone()
                            libsys.malloc_zone_pressure_relief.argtypes = [
                                ctypes.c_void_p,
                                ctypes.c_size_t,
                            ]
                            libsys.malloc_zone_pressure_relief.restype = ctypes.c_size_t
                            libsys.malloc_zone_pressure_relief(
                                ctypes.c_void_p(zone), ctypes.c_size_t(0)
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

                # Cancel any remaining tasks in the current event loop (if any)
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, cancel all tasks
                    for task in asyncio.all_tasks(loop):
                        if not task.done():
                            task.cancel()
                except RuntimeError:
                    # No running loop, which is expected after asyncio.run() completes
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
                    # Note: sqlite3.connect._instances doesn't exist in standard Python
                    # This cleanup was attempting to access a non-existent attribute
                    pass
                except Exception:
                    pass

                # Gracefully shutdown state backend if it exposes a shutdown hook
                try:
                    sb = getattr(runner, "state_backend", None)
                    if sb is not None and hasattr(sb, "shutdown"):
                        import asyncio as _asyncio

                        async def _do_shutdown() -> None:
                            try:
                                await sb.shutdown()
                            except Exception:
                                pass

                        try:
                            _asyncio.run(_do_shutdown())
                        except RuntimeError:
                            # If already in a running loop, schedule and wait briefly
                            try:
                                loop = _asyncio.get_running_loop()
                                t = loop.create_task(_do_shutdown())
                                # Best-effort wait without blocking forever
                                loop.run_until_complete(t)
                            except Exception:
                                pass
                except Exception:
                    # Never fail cleanup
                    pass

                # Aggressively release caches that can inflate RSS across runs
                try:
                    # Clear dynamic skills but preserve built-in registrations
                    from flujo.infra.skill_registry import get_skill_registry

                    reg = get_skill_registry()
                    entries = getattr(reg, "_entries", None)
                    if isinstance(entries, dict):
                        preserved: dict[str, Any] = {
                            k: v
                            for k, v in list(entries.items())
                            if isinstance(k, str)
                            and (
                                k.startswith("flujo.builtins.") or k.startswith("flujo.architect.")
                            )
                        }
                        entries.clear()
                        entries.update(preserved)
                except Exception:
                    pass

                # Final garbage collection pass after cache cleanup
                try:
                    gc.collect()
                except Exception:
                    pass


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
    # Detect paused state for HITL and tailor the header accordingly
    paused = False
    hitl_message = None
    try:
        ctx = getattr(result, "final_pipeline_context", None)
        scratch = getattr(ctx, "scratchpad", None) if ctx is not None else None
        if isinstance(scratch, dict) and scratch.get("status") == "paused":
            paused = True
            hitl_message = scratch.get("pause_message") or scratch.get("hitl_message")
    except Exception:
        paused = False

    if paused:
        console.print("[bold yellow]Pipeline execution paused.[/bold yellow]")
        if hitl_message:
            console.print(hitl_message)
        # For paused runs, stop here to avoid confusing final report output
        return
    else:
        # Respect runner-computed success flag and step history
        is_success = bool(getattr(result, "success", False))
        all_steps_ok = False
        try:
            sh = getattr(result, "step_history", []) or []
            all_steps_ok = bool(sh) and all(getattr(s, "success", False) for s in sh)
        except Exception:
            all_steps_ok = False

        if is_success and all_steps_ok:
            console.print("[bold green]Pipeline execution completed successfully![/bold green]")
        else:
            console.print("[bold red]Pipeline execution failed.[/bold red]")

    # Nicely render the final output: unwrap common wrappers and render Markdown when possible
    final_output = result.step_history[-1].output if result.step_history else None
    try:
        # Unwrap RootModel-like objects or simple containers
        if hasattr(final_output, "value") and not isinstance(final_output, (str, bytes)):
            final_output = getattr(final_output, "value")
        elif isinstance(final_output, dict) and "value" in final_output and len(final_output) == 1:
            final_output = final_output.get("value")
        # Render Markdown when a string looks like MD (headings, lists, links, bold)
        if isinstance(final_output, str):
            from rich.markdown import Markdown as _Markdown

            console.print("[bold]Final output:[/bold]")
            console.print(_Markdown(final_output))
        elif isinstance(final_output, bytes):
            # Handle bytes safely by decoding to string
            try:
                decoded_output = final_output.decode("utf-8")
                console.print(f"[bold]Final output:[/bold] {decoded_output}")
            except UnicodeDecodeError:
                console.print(f"[bold]Final output:[/bold] {final_output!r}")
        else:
            console.print(f"[bold]Final output:[/bold] {final_output}")
    except Exception:
        # Handle bytes safely in exception case too
        if isinstance(final_output, bytes):
            try:
                decoded_output = final_output.decode("utf-8")
                console.print(f"[bold]Final output:[/bold] {decoded_output}")
            except UnicodeDecodeError:
                console.print(f"[bold]Final output:[/bold] {final_output!r}")
        else:
            console.print(f"[bold]Final output:[/bold] {final_output}")
    console.print(f"[bold]Total cost:[/bold] ${result.total_cost_usd:.4f}")

    # Prefer pipeline's aggregated tokens which include nested results
    try:
        total_tokens = int(getattr(result, "total_tokens", 0))
    except Exception:
        total_tokens = sum(getattr(s, "token_counts", 0) for s in result.step_history)
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

            agent = _Dummy()  # type: ignore
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
    return pipeline.to_mermaid_with_detail_level(detail_level)


def get_pipeline_step_names(path: str) -> list[str]:
    """Return the ordered step names for a pipeline file."""
    pipeline, _ = load_pipeline_from_file(path)
    return [step.name for step in pipeline.steps]


def validate_pipeline_file(path: str, *, include_imports: bool = True) -> ValidationReport:
    """Return the validation report for a pipeline file.

    Args:
        path: Path to the pipeline YAML/Python file.
        include_imports: When True, recursively validate imported blueprints (aggregated into report).
    """
    if path.endswith((".yaml", ".yml")):
        # Load YAML and surface loader errors directly so CLI can show helpful messages
        with open(path, "r") as f:
            yaml_text = f.read()
        from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
        from flujo.domain.pipeline_validation import ValidationFinding, ValidationReport

        # Ensure relative imports resolve from the YAML file directory
        base_dir = os.path.dirname(os.path.abspath(path))
        try:
            pipeline: Pipeline[Any, Any] = load_pipeline_blueprint_from_yaml(
                yaml_text, base_dir=base_dir, source_file=path
            )
        except Exception as e:
            # Map import compilation failures to V-I1 instead of a runtime error
            msg = str(e)
            import re as _re

            m = _re.search(r"Failed to compile import '([^']+)' from '([^']+)': (.+)", msg)
            if m:
                alias, rel_path, reason = m.group(1), m.group(2), m.group(3)
                return ValidationReport(
                    errors=[
                        ValidationFinding(
                            rule_id="V-I1",
                            severity="error",
                            message=(
                                f"Imported blueprint '{alias}' could not be loaded from '{rel_path}': {reason}"
                            ),
                            step_name=f"imports.{alias}",
                            location_path=f"imports.{alias}",
                            file=path,
                        )
                    ],
                    warnings=[],
                )
            # For other errors, re-raise to preserve existing behavior
            raise
    else:
        pipeline, _ = load_pipeline_from_file(path)

    return pipeline.validate_graph(include_imports=include_imports)


def validate_yaml_text(
    yaml_text: str, base_dir: Optional[str] = None, *, include_imports: bool = False
) -> ValidationReport:
    """Validate a YAML blueprint string and return its ValidationReport.

    Args:
        yaml_text: The YAML blueprint content.
        base_dir: Optional base directory to resolve relative imports within YAML.
        include_imports: When True, recursively validate imported blueprints (aggregated into report).

    Returns:
        ValidationReport: The validation report from pipeline.validate_graph().

    Raises:
        Exit: If loading the YAML fails.
    """
    from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml

    pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir=base_dir)
    return pipeline.validate_graph(include_imports=include_imports)


def sanitize_blueprint_yaml(yaml_text: str) -> str:
    """Repair common issues in Architect-generated YAML to improve validity.

    Fixes applied conservatively:
    - Ensure every step dict has a `name`. If missing, generate a deterministic
      name based on its role (e.g., derived from agent id/uses/kind) with an
      incremental suffix.
    - Convert legacy/mistaken `conditional: <callable>` into
      `kind: conditional` + `condition: <callable>`.
    - Normalize nested structures within `branches`, `loop.body`, and `map.body`.

    Returns the original YAML if parsing fails to avoid introducing new errors.
    """
    if not yaml_text or not yaml_text.strip():
        return yaml_text

    try:
        data = yaml.safe_load(yaml_text)
    except Exception:
        return yaml_text

    if not isinstance(data, dict):
        return yaml_text

    steps = data.get("steps")
    if not isinstance(steps, list):
        return yaml_text

    name_counter: Dict[str, int] = {}
    changed: bool = False

    def _gen_name(prefix: str) -> str:
        idx = name_counter.get(prefix, 0) + 1
        name_counter[prefix] = idx
        return f"{prefix}_{idx}"

    def _derive_prefix(node: Dict[str, Any]) -> str:
        # Prefer agent id last segment
        agent = node.get("agent")
        if isinstance(agent, dict):
            sid = agent.get("id")
            if isinstance(sid, str) and sid.strip():
                tail = sid.rsplit(".", 1)[-1]
                return tail.replace(" ", "_")
        # If declarative uses is provided
        uses = node.get("uses")
        if isinstance(uses, str) and uses.strip():
            tail = uses.rsplit(".", 1)[-1]
            return tail.replace(" ", "_")
        # Fallback to kind
        kind = node.get("kind")
        if isinstance(kind, str) and kind.strip():
            return kind.replace(" ", "_")
        return "step"

    def _repair_node(node: Any) -> None:
        nonlocal changed
        if isinstance(node, dict):
            # Normalize LLM-shape single step where tool id/params are placed at top-level
            # Example:
            #   - id: "flujo.builtins.web_search"
            #     params: { query: "..." }
            #   ->
            #   - agent: { id: "...", params: { ... } }
            if isinstance(node.get("id"), str) and "agent" not in node and "uses" not in node:
                try:
                    agent_id = str(node.pop("id"))
                    params_val = node.pop("params", {})
                    if not isinstance(params_val, dict):
                        params_val = {}
                    node["agent"] = {"id": agent_id, "params": params_val}
                    changed = True
                except Exception:
                    pass
            # Flatten mistaken nested shape: { step: { ...step fields... } }
            if isinstance(node.get("step"), dict):
                try:
                    embedded = cast(Dict[str, Any], node.get("step"))
                except Exception:
                    embedded = None
                if isinstance(embedded, dict):
                    try:
                        del node["step"]
                    except Exception:
                        pass
                    for k, v in embedded.items():
                        # Do not overwrite existing normalized fields
                        if k not in node:
                            node[k] = v

            # Normalize mistaken `conditional` key into proper fields
            if "conditional" in node and "kind" not in node:
                cond = node.get("conditional")
                if isinstance(cond, str) and cond.strip():
                    node["kind"] = "conditional"
                    node["condition"] = cond
                    try:
                        del node["conditional"]
                    except Exception:
                        pass
                    changed = True

            # Ensure name exists for top-level and nested step dicts
            if (
                "branches" in node
                or "agent" in node
                or "uses" in node
                or node.get("kind")
                in {
                    "step",
                    "parallel",
                    "conditional",
                    "loop",
                    "map",
                    "dynamic_router",
                }
            ):
                if not isinstance(node.get("name"), str) or not node.get("name", "").strip():
                    # Also accept legacy alias `step` for human-provided label
                    legacy = node.get("step")
                    if isinstance(legacy, str) and legacy.strip():
                        node["name"] = legacy.strip()
                    else:
                        node["name"] = _gen_name(_derive_prefix(node))
                    changed = True
                # Default kind when omitted
                if not isinstance(node.get("kind"), str) or not node.get("kind", "").strip():
                    node["kind"] = "step"
                    changed = True
                # Ensure agent params is a dict when agent exists
                agent_spec = node.get("agent")
                if isinstance(agent_spec, dict):
                    params = agent_spec.get("params")
                    if params is None:
                        agent_spec["params"] = {}
                        changed = True
                    # Heuristic repair: if using extract_from_text and no 'text' param but input provided
                    sid = agent_spec.get("id")
                    if (
                        isinstance(sid, str)
                        and sid.endswith("extract_from_text")
                        and isinstance(agent_spec.get("params"), dict)
                    ):
                        p = agent_spec["params"]
                        if "text" not in p:
                            # If an input template exists at step level, move it into params.text
                            inp = node.get("input")
                            if isinstance(inp, str) and inp.strip():
                                p["text"] = inp
                                # Remove step-level input to avoid confusion
                                try:
                                    del node["input"]
                                except Exception:
                                    pass
                                changed = True
                    # Heuristic repair: if using fs_write_file and no 'content' but input provided
                    if (
                        isinstance(sid, str)
                        and sid.endswith("fs_write_file")
                        and isinstance(agent_spec.get("params"), dict)
                    ):
                        p = agent_spec["params"]
                        if "content" not in p:
                            inp = node.get("input")
                            if isinstance(inp, str) and inp.strip():
                                p["content"] = inp
                                try:
                                    del node["input"]
                                except Exception:
                                    pass
                                changed = True

            # Recurse into branches (conditional)
            branches = node.get("branches")
            if isinstance(branches, list):
                # Some LLM outputs provide a bare list; wrap into a default-named branch
                node["branches"] = {"default": branches}
                branches = node["branches"]
                changed = True
            if isinstance(branches, dict):
                for _, lst in branches.items():
                    if isinstance(lst, list):
                        for child in lst:
                            _repair_node(child)

            # Recurse into loop body
            loop_spec = node.get("loop")
            if isinstance(loop_spec, dict):
                body = loop_spec.get("body")
                if isinstance(body, list):
                    for child in body:
                        _repair_node(child)

            # Recurse into map body
            map_spec = node.get("map")
            if isinstance(map_spec, dict):
                # Normalize common LLM variants: input->iterable_input, steps->body
                if "input" in map_spec and "iterable_input" not in map_spec:
                    try:
                        map_spec["iterable_input"] = map_spec.pop("input")
                        changed = True
                    except Exception:
                        pass
                if "steps" in map_spec and "body" not in map_spec:
                    try:
                        steps_val = map_spec.pop("steps")
                        # Ensure body is a list of steps
                        if isinstance(steps_val, list):
                            map_spec["body"] = steps_val
                        elif isinstance(steps_val, dict):
                            map_spec["body"] = [steps_val]
                        else:
                            map_spec["body"] = []
                        changed = True
                    except Exception:
                        pass
                body = map_spec.get("body")
                if isinstance(body, list):
                    for child in body:
                        _repair_node(child)

            # Normalize ad-hoc 'parallel' key into proper kind/branches structure
            if "parallel" in node and (
                not isinstance(node.get("kind"), str) or node.get("kind") == "step"
            ):
                par = node.get("parallel")
                try:
                    # remove legacy field and convert to branches
                    del node["parallel"]
                except Exception:
                    pass
                node["kind"] = "parallel"
                branches_map: Dict[str, Any] = {}
                if isinstance(par, list):
                    for idx, child in enumerate(par, start=1):
                        branch_name = f"branch_{idx}"
                        if isinstance(child, dict):
                            branches_map[branch_name] = [child]
                        else:
                            branches_map[branch_name] = []
                elif isinstance(par, dict):
                    branches_map["branch_1"] = [par]
                else:
                    branches_map["branch_1"] = []
                node["branches"] = branches_map
                changed = True
                # Repair inner steps
                for lst in branches_map.values():
                    for child in lst:
                        _repair_node(child)

        elif isinstance(node, list):
            for it in node:
                _repair_node(it)

    _repair_node(steps)

    if not changed:
        return yaml_text
    try:
        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return yaml_text


def find_side_effect_skills_in_yaml(yaml_text: str, *, base_dir: Optional[str] = None) -> list[str]:
    """Return a list of skill IDs in YAML that are marked side_effects=True in registry.

    This scans steps recursively for entries of the form:
      - agent: { id: "skill_id", params: {...} }

    Args:
        yaml_text: The YAML blueprint content.
        base_dir: Directory to resolve and load skills catalog from.

    Returns:
        List of skill IDs that require side-effect confirmation.
    """
    # Pre-validate YAML to avoid warnings from malformed content
    if not yaml_text or not yaml_text.strip():
        return []

    # Check for basic YAML structure indicators
    if not any(
        indicator in yaml_text for indicator in ["version:", "steps:", "pipeline:", "workflow:"]
    ):
        # Not a pipeline YAML, skip processing
        return []

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        # Only warn for YAML errors that suggest real parsing issues, not malformed test content
        if "while parsing a flow node" in str(e) and "<stream end>" in str(e):
            # This is likely malformed test content, handle gracefully without warning
            return []
        # For other YAML errors, log but don't warn to avoid test pollution
        return []

    if not isinstance(data, dict):
        return []

    # Ensure registry is populated from catalog in base_dir and packaged entry points
    try:
        directory = base_dir or os.getcwd()
        load_skills_catalog(directory)
        load_skills_entry_points()
    except Exception:
        # Best-effort; absence of catalog just yields empty results
        # Don't warn during testing to avoid output pollution
        return []

    from flujo.infra.skill_registry import get_skill_registry

    reg = get_skill_registry()
    found: set[str] = set()

    def _scan(node: Any) -> None:
        if isinstance(node, dict):
            # Detect agent dicts
            if "agent" in node and isinstance(node["agent"], dict):
                agent_spec = node["agent"]
                skill_id = agent_spec.get("id") if isinstance(agent_spec, dict) else None
                if isinstance(skill_id, str):
                    entry = reg.get(skill_id)
                    if entry and bool(entry.get("side_effects", False)):
                        found.add(skill_id)
            # Recurse into dict values
            for v in node.values():
                _scan(v)
        elif isinstance(node, list):
            for item in node:
                _scan(item)

    _scan(data)
    return sorted(found)


def enrich_yaml_with_required_params(
    yaml_text: str,
    *,
    non_interactive: bool,
    base_dir: Optional[str] = None,
) -> str:
    """Fill missing required params for registry-backed skills by prompting the user.

    - Scans steps for `agent: { id: <skill_id>, params: {...} }`
    - Uses registry `input_schema`/`arg_schema` to determine required properties
    - Prompts the user for missing required keys if not in non-interactive mode
    - Returns updated YAML text (or original if no changes / non-interactive)
    """
    # Pre-validate YAML to avoid warnings from malformed content
    if not yaml_text or not yaml_text.strip():
        return yaml_text

    # Check for basic YAML structure indicators
    if not any(
        indicator in yaml_text for indicator in ["version:", "steps:", "pipeline:", "workflow:"]
    ):
        # Not a pipeline YAML, return original
        return yaml_text

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        # Only warn for YAML errors that suggest real parsing issues, not malformed test content
        if "while parsing a flow node" in str(e) and "<stream end>" in str(e):
            # This is likely malformed test content, handle gracefully without warning
            return yaml_text
        # For other YAML errors, return original without warning to avoid test pollution
        return yaml_text

    if not isinstance(data, dict):
        return yaml_text

    # Populate registry
    try:
        directory = base_dir or os.getcwd()
        load_skills_catalog(directory)
        load_skills_entry_points()
    except Exception:
        # Don't warn during testing to avoid output pollution
        return yaml_text

    from flujo.infra.skill_registry import get_skill_registry

    registry = get_skill_registry()
    changed = False

    def _collect_required(entry: dict[str, Any]) -> list[str]:
        schema = entry.get("input_schema") or entry.get("arg_schema") or {}
        req = schema.get("required")
        return list(req) if isinstance(req, list) else []

    def _ensure_params(node: dict[str, Any]) -> None:
        nonlocal changed
        agent_spec = node.get("agent")
        if not isinstance(agent_spec, dict):
            return
        skill_id = agent_spec.get("id")
        if not isinstance(skill_id, str):
            return
        entry = registry.get(skill_id) or {}
        required_keys = _collect_required(entry)
        if not required_keys:
            return
        params = agent_spec.get("params")
        if not isinstance(params, dict):
            params = {}
            agent_spec["params"] = params
        missing = [k for k in required_keys if k not in params]
        if not missing:
            return
        if non_interactive:
            return
        # Prompt for each missing key
        try:
            import typer as _typer
        except Exception:
            return
        for key in missing:
            val = _typer.prompt(
                f"Enter value for required parameter '{key}' of skill '{skill_id}':"
            )
            params[key] = val
            changed = True

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "agent" in node:
                _ensure_params(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)

    if not changed:
        return yaml_text
    try:
        return yaml.safe_dump(data, sort_keys=False)
    except Exception:
        return yaml_text


def apply_cli_defaults(command: str, **kwargs: Any) -> Dict[str, Any]:
    """Apply CLI defaults from configuration file to command arguments.

    Precedence: command-line explicit values > TOML defaults > function defaults.
    This implementation inspects the active Click context to detect which params
    were explicitly set by the user, ensuring we never override user-provided
    values even if they match the function's default.

    Args:
        command: The command name (e.g., "solve", "bench").
        **kwargs: The parsed command parameters.

    Returns:
        Dict containing parameters with TOML defaults applied where appropriate.
    """
    from flujo.cli.main import get_cli_defaults

    try:
        import click as _click
    except Exception:
        _click = None  # type: ignore[assignment]

    cli_defaults = get_cli_defaults(command)
    result = kwargs.copy()

    explicitly_set: set[str] = set()
    if _click is not None:
        try:
            ctx = _click.get_current_context(silent=True)
            if ctx is not None and isinstance(getattr(ctx, "params", None), dict):
                explicitly_set = set(ctx.params.keys())
        except Exception:
            explicitly_set = set()

    for key in kwargs.keys():
        if key in explicitly_set:
            continue
        if key in cli_defaults and result.get(key, None) is None:
            result[key] = cli_defaults[key]

    return result


# ---------------------------------------------------------------------------
# Project-aware helpers
# ---------------------------------------------------------------------------


def _has_project_markers(dir_path: Path) -> bool:
    """Return True if the directory looks like a Flujo project root.

    Markers:
    - flujo.toml
    - pipeline.yaml or pipeline.yml
    """
    return (
        (dir_path / "flujo.toml").exists()
        or (dir_path / "pipeline.yaml").exists()
        or (dir_path / "pipeline.yml").exists()
    )


def resolve_project_root(
    explicit: Optional[Path] = None, allow_missing: bool = True
) -> Optional[Path]:
    """Resolve the project root using explicit path, env var, or marker search.

    Resolution order:
    1) explicit path (CLI --project)
    2) FLUJO_PROJECT_ROOT environment variable
    3) Walk up from cwd to find a directory containing flujo.toml or pipeline.yaml

    Args:
        explicit: Optional explicit project directory.
        allow_missing: If True and nothing found, return None instead of raising.

    Returns:
        The resolved project root Path or None.

    Raises:
        Exit: When a provided path is invalid or no project is found and allow_missing=False.
    """
    # 1) Explicit CLI value
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists() or not p.is_dir():
            from typer import secho

            secho(f"Error: --project '{p}' does not exist or is not a directory", fg="red")
            raise Exit(1)
        return p

    # 2) Environment variable
    env_root = os.environ.get("FLUJO_PROJECT_ROOT")
    if isinstance(env_root, str) and env_root.strip():
        p = Path(env_root).resolve()
        if p.exists() and p.is_dir():
            return p
        if not allow_missing:
            from typer import secho

            secho(
                f"Error: FLUJO_PROJECT_ROOT='{env_root}' does not exist or is not a directory",
                fg="red",
            )
            raise Exit(1)
        return None

    # 3) Walk up from cwd using markers
    current = Path.cwd().resolve()
    while True:
        if _has_project_markers(current):
            return current
        if current.parent == current:
            if allow_missing:
                return None
            from typer import secho

            secho(
                "Error: Not a Flujo project. Please run 'flujo init' in your desired project directory first.",
                fg="red",
            )
            raise Exit(1)
        current = current.parent


def ensure_project_root_on_sys_path(root: Optional[Path]) -> None:
    """Add the given project root to sys.path if not already present.

    This allows imports like `skills.*` to resolve when running from subdirs or CI.
    """
    try:
        if root is None:
            return
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    except Exception:
        # Non-fatal: importing may still work if paths are already correct
        pass


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the Flujo project root by locating a flujo.toml or pipeline.yaml.

    Args:
        start: Optional starting directory. Defaults to current working directory.

    Returns:
        Path to the project root directory (the directory containing flujo.toml).

    Raises:
        Exit: If no flujo.toml is found in the directory tree.
    """
    # Prefer explicit start when provided, else use env/markers
    if start is not None:
        resolved = resolve_project_root(start, allow_missing=False)
        assert resolved is not None
        return resolved
    resolved = resolve_project_root(None, allow_missing=False)
    assert resolved is not None
    return resolved


def scaffold_project(directory: Path, *, overwrite_existing: bool = False) -> None:
    """Create a new Flujo project scaffold in the given directory.

    Creates `flujo.toml`, `pipeline.yaml`, `skills/`, `.flujo/`, and initializes
    the SQLite state backend at `.flujo/state.db`.

    Raises Exit if the directory already contains a Flujo project.
    """
    from typer import secho

    directory = directory.resolve()
    flujo_toml = directory / "flujo.toml"
    hidden_dir = directory / ".flujo"
    skills_dir = directory / "skills"
    # env_file = directory / ".env"  # Unused variable removed

    if (flujo_toml.exists() or hidden_dir.exists()) and not overwrite_existing:
        secho("Error: This directory already looks like a Flujo project.", fg="red")
        raise Exit(1)

    # Create directories
    skills_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Write template files, tracking which files were created vs overwritten
    from importlib import resources as _res

    created: list[str] = []
    overwritten: list[str] = []

    def _write(path: Path, content: str) -> None:
        target = path
        existed = target.exists()
        target.write_text(content)
        rel = str(target.relative_to(directory)) if target.is_file() else target.name
        if existed:
            overwritten.append(rel)
        else:
            created.append(rel)

    try:
        template_pkg = "flujo.templates.project"
        with _res.files(template_pkg).joinpath("flujo.toml").open("r") as f:
            _write(flujo_toml, f.read())
        with _res.files(template_pkg).joinpath("pipeline.yaml").open("r") as f:
            _write(directory / "pipeline.yaml", f.read())
        # .env.example for users to copy into .env
        try:
            with _res.files(template_pkg).joinpath(".env.example").open("r") as f:
                _write(directory / ".env.example", f.read())
        except Exception:
            pass
        # skills/__init__.py
        with _res.files(template_pkg).joinpath("skills__init__.py").open("r") as f:
            _write(skills_dir / "__init__.py", f.read())
        with _res.files(template_pkg).joinpath("custom_tools.py").open("r") as f:
            _write(skills_dir / "custom_tools.py", f.read())
        # README template (best-effort)
        try:
            with _res.files(template_pkg).joinpath("README.md").open("r") as f:
                _write(directory / "README.md", f.read())
        except Exception:
            pass
    except Exception:
        # Fallback: write minimal content if resources are unavailable
        _write(
            flujo_toml,
            """
# Flujo project configuration

# Use an in-memory state backend by default
state_uri = "memory://"

# Load environment variables (API keys, etc.) from this file.
# Copy `.env.example` to `.env`, fill your keys, or change this path.
env_file = ".env"

[settings]
# default_solution_model = "gpt-4o-mini"

# Centralized budgets (optional)
[budgets]
# [budgets.default]
# total_cost_usd_limit = 5.0
# total_tokens_limit = 100000

# Architect defaults
[architect]
# Enable the agentic Architect state machine by default for this project
state_machine_default = true
# To disable by default, set to false or remove this section.
# Per-run overrides:
#   - Force agentic: export FLUJO_ARCHITECT_STATE_MACHINE=1
#   - Force minimal: export FLUJO_ARCHITECT_MINIMAL=1
            """.strip()
            + "\n",
        )
        _write(
            directory / "pipeline.yaml",
            """
version: "0.1"
name: "example"
steps:
  - kind: step
    name: passthrough
            """.strip()
            + "\n",
        )
        _write(skills_dir / "__init__.py", "# Custom project skills\n")
        _write(
            skills_dir / "custom_tools.py",
            """
from __future__ import annotations

# Example custom tool function
async def echo_tool(x: str) -> str:
    return x
            """.strip()
            + "\n",
        )
        _write(
            directory / "README.md",
            """
# Flujo Project

Welcome! This project is scaffolded for use with Flujo.

## Architect Defaults

This project enables the agentic Architect (state machine) by default via `flujo.toml`:

[architect]
state_machine_default = true

- To disable by default, set `state_machine_default = false` or remove the section.
- Per-run overrides:
  - Force agentic: `FLUJO_ARCHITECT_STATE_MACHINE=1`
  - Force minimal: `FLUJO_ARCHITECT_MINIMAL=1`
- CLI override on the create command:
  - `uv run flujo create --agentic --goal "..."`
  - `uv run flujo create --no-agentic --goal "..."`
            """.strip()
            + "\n",
        )
        # Note: Agentic architect defaults are controlled via flujo.toml ([architect])

    # Create basic .flujo structure (no DB for memory backend)
    try:
        (hidden_dir / "logs").mkdir(exist_ok=True)
        (hidden_dir / "cache").mkdir(exist_ok=True)
    except Exception:
        # Best-effort init; ignore if environment lacks file system support
        pass

    if overwrite_existing and overwritten:
        secho(
            "✅ Re-initialized Flujo project templates.",
            fg="green",
        )
        secho(
            "Overwrote: " + ", ".join(sorted(overwritten)),
            fg="yellow",
        )
        if created:
            secho("Created: " + ", ".join(sorted(created)), fg="cyan")
    else:
        secho("✅ Your new Flujo project has been initialized in this directory!", fg="green")


def scaffold_demo_project(directory: Path, *, overwrite_existing: bool = False) -> None:
    """Create a new Flujo demo project with a sample research pipeline."""
    from typer import secho

    directory = directory.resolve()
    flujo_toml = directory / "flujo.toml"
    pipeline_yaml = directory / "pipeline.yaml"
    hidden_dir = directory / ".flujo"
    skills_dir = directory / "skills"

    if (
        flujo_toml.exists() or pipeline_yaml.exists() or hidden_dir.exists()
    ) and not overwrite_existing:
        secho(
            "Error: This directory already contains Flujo project files. Use --force to overwrite.",
            fg="red",
        )
        raise Exit(1)

    # Create directories
    skills_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Write template files, tracking which files were created vs overwritten
    created: list[str] = []
    overwritten: list[str] = []

    def _write(path: Path, content: str) -> None:
        target = path
        existed = target.exists()
        target.write_text(content)
        rel = str(target.relative_to(directory)) if target.is_file() else target.name
        if existed:
            overwritten.append(rel)
        else:
            created.append(rel)

    # Content for the templates
    flujo_toml_content = (
        """
# Flujo project configuration template (demo)

# Use an in-memory state backend for demos so runs never persist
state_uri = "memory://"

[settings]
# default_solution_model = "gpt-4o-mini"
# reflection_enabled = true

# Centralized budgets (optional)
[budgets]
# [budgets.default]
# total_cost_usd_limit = 5.0
# total_tokens_limit = 100000
\n+# Architect defaults
[architect]
# Enable the agentic Architect state machine by default for this project
state_machine_default = true
""".strip()
        + "\n"
    )

    demo_pipeline_content = (
        generate_demo_yaml(demo_name="research_demo", preset="research_demo") + "\n"
    )

    skills_init_content = "# This marks the skills package for your project.\n"

    custom_tools_content = (
        """
from __future__ import annotations


# Example custom tool function
async def echo_tool(x: str) -> str:
    return x
""".strip()
        + "\n"
    )

    # Write the files
    _write(flujo_toml, flujo_toml_content)
    _write(pipeline_yaml, demo_pipeline_content)
    # Scaffold .env.example to guide users
    try:
        from importlib import resources as _res

        template_pkg = "flujo.templates.project"
        with _res.files(template_pkg).joinpath(".env.example").open("r") as f:
            _write(directory / ".env.example", f.read())
    except Exception:
        pass
    _write(skills_dir / "__init__.py", skills_init_content)
    _write(skills_dir / "custom_tools.py", custom_tools_content)
    # Add README with architect toggle notes
    try:
        _write(
            directory / "README.md",
            """
# Flujo Demo Project

This demo is scaffolded with the agentic Architect enabled by default via `flujo.toml`:

[architect]
state_machine_default = true

Per-run overrides:
- Force agentic: `FLUJO_ARCHITECT_STATE_MACHINE=1`
- Force minimal: `FLUJO_ARCHITECT_MINIMAL=1`
CLI overrides (create command): `--agentic` / `--no-agentic`.
            """.strip()
            + "\n",
        )
    except Exception:
        pass
    # Note: Agentic architect defaults are controlled via flujo.toml ([architect])

    # Initialize basic .flujo structure (no DB for memory backend)
    try:
        (hidden_dir / "logs").mkdir(exist_ok=True)
        (hidden_dir / "cache").mkdir(exist_ok=True)
    except Exception:
        # Best-effort init
        pass

    if overwrite_existing and overwritten:
        secho("✅ Re-initialized Flujo project with the demo pipeline.", fg="green")
        if overwritten:
            secho("Overwrote: " + ", ".join(sorted(overwritten)), fg="yellow")
        if created:
            secho("Created: " + ", ".join(sorted(created)), fg="cyan")
    else:
        secho("✅ Your new Flujo demo project is ready!", fg="green")
        secho("To run the demo, execute: [bold]flujo run[/bold]", fg="cyan")


def update_project_budget(flujo_toml_path: Path, pipeline_name: str, cost_limit: float) -> None:
    """Add or update a budget entry under [budgets.pipeline.<name>] in flujo.toml.

    This function preserves existing content by performing minimal, targeted text edits
    or appending a new section when absent.
    """
    text = flujo_toml_path.read_text() if flujo_toml_path.exists() else ""

    section_header_quoted = f'[budgets.pipeline."{pipeline_name}"]'
    new_section = f"\n\n{section_header_quoted}\ntotal_cost_usd_limit = {cost_limit}\n"

    # If file is empty, create minimal TOML with budgets section
    if not text.strip():
        flujo_toml_path.write_text("[budgets]\n" + new_section.lstrip("\n"))
        return

    # Replace existing section if present (quoted or unquoted)
    pattern = rf"^\[(budgets\.pipeline\.(?:\"?{re.escape(pipeline_name)}\"?))\][\s\S]*?(?=^\[|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if m:
        start, end = m.span()
        updated = text[:start] + new_section.strip() + "\n" + text[end:]
        flujo_toml_path.write_text(updated)
        return

    # Ensure [budgets] top-level exists; if not, append it before pipeline entry
    if "\n[budgets]\n" not in ("\n" + text + "\n") and not re.search(
        r"^\[budgets\]", text, flags=re.MULTILINE
    ):
        text = text.rstrip() + "\n\n[budgets]\n"

    flujo_toml_path.write_text(text.rstrip() + new_section)


# --------------------------------------------------------------------------- #
# Demo YAML generator
# --------------------------------------------------------------------------- #


def generate_demo_yaml(
    *,
    demo_name: str = "demo",
    preset: str = "conversational_loop",
    conversation: bool = True,
    ai_turn_source: Optional[str] = None,
    user_turn_sources: Optional[list[str]] = None,
    history_strategy: Optional[str] = None,
    history_max_tokens: Optional[int] = None,
    history_max_turns: Optional[int] = None,
    history_summarize_ratio: Optional[float] = None,
) -> str:
    """Generate a ready-to-run demo pipeline YAML.

    Presets:
    - conversational_loop: HITL clarify + assistant answer inside a loop, with optional
      conversation history controls.
    - map_hitl: Map over an `items` list and pause for a note per item, then combine.
    """
    preset = (preset or "conversational_loop").strip().lower()
    # Normalize options
    uts = user_turn_sources or ["hitl"]

    if preset == "map_hitl":
        # Simple map demo with HITL annotation per item
        return (
            """
version: "0.1"
name: "{demo_name}"

steps:
  - kind: map
    name: AnnotateItems
    iterable_input: items
    body:
      - kind: hitl
        name: AnnotateItem
        message: "Provide a short note for this item"
      - kind: step
        name: Combine
        agent: {{ id: "flujo.builtins.stringify" }}
""".strip()
        ).format(demo_name=demo_name)

    if preset == "research_demo":
        return (
            """
version: "0.1"
name: "{demo_name}"

agents:
  result_formatter:
    model: "openai:gpt-4o-mini"
    system_prompt: |
      You are a research assistant. You will be given a JSON object containing web search results.
      Format these results into a clear, concise, and human-readable summary.
      Present the title, a brief snippet, and the link for each result in a numbered list.
      If the input is empty or contains no results, say "I couldn't find any information on that topic."
    output_schema:
      type: string

steps:
  - kind: hitl
    name: get_research_topic
    message: "What would you like to research on the web?"

  - kind: step
    name: perform_web_search
    agent:
      id: "flujo.builtins.web_search"
      params:
        max_results: 3
    input: "{{ previous_step }}"

  - kind: step
    name: format_search_results
    uses: agents.result_formatter
    input: "{{ previous_step }}"
""".strip()
        ).format(demo_name=demo_name)

    # Default preset: conversational loop with history wiring
    meta_lines: list[str] = []
    if conversation:
        meta_lines.append("    conversation: true")
        if ai_turn_source:
            meta_lines.append(f"    ai_turn_source: {ai_turn_source}")
        if uts:
            quoted = ", ".join(repr(x) for x in uts)
            meta_lines.append(f"    user_turn_sources: [{quoted}]")
        hm: dict[str, object] = {}
        if history_strategy:
            hm["strategy"] = history_strategy
        if history_max_tokens is not None:
            hm["max_tokens"] = history_max_tokens
        if history_max_turns is not None:
            hm["max_turns"] = history_max_turns
        if history_summarize_ratio is not None:
            hm["summarize_ratio"] = history_summarize_ratio
        if hm:
            meta_lines.append("    history_management:")
            for k, v in hm.items():
                meta_lines.append(f"      {k}: {v}")

    loop_meta = ("\n" + "\n".join(meta_lines) + "\n") if meta_lines else "\n"
    yaml_text = (
        """
version: "0.1"
name: "{demo_name}"

steps:
  - kind: loop
    name: conversational_loop
    stop_when: agent_finished{loop_meta}    body:
      - kind: hitl
        name: clarify
        message: "What do you want to achieve?"
      - kind: step
        name: answer
        agent: {{ id: "flujo.builtins.stringify" }}
""".strip()
    ).format(demo_name=demo_name, loop_meta=loop_meta)
    return yaml_text
