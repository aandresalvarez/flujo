"""CLI entry point for flujo."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.domain.dsl.pipeline import Pipeline
    from flujo.domain.models import Candidate, Checklist, Task
    from flujo.agents.wrapper import AsyncAgentWrapper
    from flujo.infra.settings import Settings as FlujoSettings

import typer
from typing_extensions import Annotated

from flujo.application.runner import Flujo as _Flujo
from flujo.exceptions import ConfigurationError, SettingsError
from flujo.infra import telemetry
from flujo.infra.config_manager import get_cli_defaults as _get_cli_defaults

# Import functions that tests expect to monkeypatch - these are module-level imports
# that can be properly monkeypatched in tests
from flujo.agents.recipes import (
    get_reflection_agent as _get_reflection_agent,
    make_review_agent as _make_review_agent,
    make_self_improvement_agent as _make_self_improvement_agent,
    make_solution_agent as _make_solution_agent,
    make_validator_agent as _make_validator_agent,
)
from flujo.application.eval_adapter import run_pipeline_async as _run_pipeline_async
from flujo.application.self_improvement import (
    ImprovementReport as _ImprovementReport,
    SelfImprovementAgent as _SelfImprovementAgent,
    evaluate_and_improve as _evaluate_and_improve,
)
from flujo.recipes.factories import run_default_pipeline as _run_default_pipeline
from flujo.utils.serialization import safe_deserialize as _safe_deserialize
from flujo.type_definitions.common import JSONObject

from .dev_commands_experimental import (
    add_eval_case_cmd as _add_eval_case_cmd,
    bench as _bench_cmd,
    improve as _improve_cmd,
    solve as _solve_cmd,
)
from .dev_commands_dev import (
    compile as _compile_cmd,
    pipeline_mermaid_cmd as _pipeline_mermaid_cmd,
    show_config_cmd as _show_config_cmd,
    version_cmd as _version_cmd,
)
from .dev_commands_budgets import budgets_show as _budgets_show
from .dev_commands_health import dev_health_check as _dev_health_check
from .app_registration import create_cli_app, register_all_commands
from .architect_command import create as _create_cmd
from .bootstrap import bootstrap_cli_runtime, configure_cli_logging
from .helpers import (
    apply_cli_defaults as _apply_cli_defaults,
    create_benchmark_table as _create_benchmark_table,
    create_flujo_runner as _create_flujo_runner,
    execute_improve as _execute_improve,
    execute_pipeline_with_output_handling as _execute_pipeline_with_output_handling,
    execute_solve_pipeline as _execute_solve_pipeline,
    load_pipeline_from_yaml_file as _load_pipeline_from_yaml_file,
    print_rich_or_typer,
    run_benchmark_pipeline as _run_benchmark_pipeline,
    setup_solve_command_environment as _setup_solve_command_environment,
    validate_yaml_text as _validate_yaml_text,
)
from .lens import lens_app
from .run_command import run as _run_cmd
from .status_command import status as _status_cmd
from .migrate_command import migrate as _migrate_cmd
from .validate_command import _validate_impl, validate as _validate_cmd, validate_dev
from .project_command import demo as _demo_cmd, init as _init_cmd

# Re-export Flujo after all imports to satisfy linting (E402)
Flujo = _Flujo

# Expose create for backward compatibility
create = _create_cmd
# Expose command callables for tests/backward compatibility
run = _run_cmd
status = _status_cmd
migrate = _migrate_cmd
validate = _validate_cmd
init = _init_cmd
demo = _demo_cmd
# Re-export helper functions for legacy test imports
apply_cli_defaults = _apply_cli_defaults
create_benchmark_table = _create_benchmark_table
create_flujo_runner = _create_flujo_runner
execute_improve = _execute_improve
load_pipeline_from_yaml_file = _load_pipeline_from_yaml_file
run_benchmark_pipeline = _run_benchmark_pipeline
setup_solve_command_environment = _setup_solve_command_environment
execute_pipeline_with_output_handling = _execute_pipeline_with_output_handling
execute_solve_pipeline = _execute_solve_pipeline
validate_yaml_text = _validate_yaml_text

# Type definitions for CLI
WeightsType = list[dict[str, Union[str, float]]]
MetadataType = JSONObject
ScorerType = (
    str  # Changed from Literal["ratio", "weighted", "reward"] to str for typer compatibility
)

# Create and configure the CLI app
app = create_cli_app()
register_all_commands(app)

# Get logfire reference for main callback
logfire = telemetry.logfire

# Expose developer commands for tests that import from flujo.cli.main
dev_health_check = _dev_health_check
solve = _solve_cmd
bench = _bench_cmd
version_cmd = _version_cmd
show_config_cmd = _show_config_cmd
add_eval_case_cmd = _add_eval_case_cmd
improve = _improve_cmd
compile = _compile_cmd
pipeline_mermaid_cmd = _pipeline_mermaid_cmd
budgets_show = _budgets_show


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
    project: Annotated[
        Optional[str],
        typer.Option(
            "--project",
            help=(
                "Project root directory (overrides FLUJO_PROJECT_ROOT). "
                "Adds it to PYTHONPATH for imports like skills.*"
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed error traces for debugging",
        ),
    ] = False,
    trace: Annotated[
        bool,
        typer.Option(
            "--trace",
            help="Alias for --verbose to print full Python tracebacks",
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

    bootstrap_cli_runtime(project_root=project, register_builtins=True, enable_telemetry=True)

    # Configure logging based on flags
    configure_cli_logging(debug=debug, verbose=verbose, trace=trace)


# Explicit exports
__all__ = [
    "app",
    "run",
    "status",
    "migrate",
    "validate",
    "_validate_impl",
    "validate_dev",
    "dev_health_check",
    "solve",
    "bench",
    "version_cmd",
    "show_config_cmd",
    "init",
    "demo",
    "create",
    "compile",
    "pipeline_mermaid_cmd",
    "lens_app",
    "apply_cli_defaults",
    "create_benchmark_table",
    "create_flujo_runner",
    "execute_improve",
    "load_pipeline_from_yaml_file",
    "run_benchmark_pipeline",
    "setup_solve_command_environment",
    "execute_pipeline_with_output_handling",
    "execute_solve_pipeline",
    "validate_yaml_text",
    "main",
]

# Conditionally register experimental commands to avoid breaking CLI tests
try:
    if os.environ.get("FLUJO_ENABLE_DEMO_YAML") == "1":
        # Note: demo_yaml_cmd would need to be imported if it exists
        pass
except Exception:
    pass


if __name__ == "__main__":
    try:
        # Local alias shim: support legacy `--format` for validate commands when executed directly
        try:
            argv = list(sys.argv)
            if any(
                tok == "--format" or (isinstance(tok, str) and tok.startswith("--format="))
                for tok in argv
            ):
                # Map for validate
                if "validate" in argv or (
                    len(argv) >= 3 and argv[1] == "dev" and argv[2] == "validate"
                ):
                    for i, tok in enumerate(argv):
                        if tok == "--format":
                            argv[i] = "--output-format"
                        elif isinstance(tok, str) and tok.startswith("--format="):
                            argv[i] = tok.replace("--format=", "--output-format=", 1)
                    sys.argv[:] = argv
                # No special mapping for other commands
        except Exception:
            pass
        app()
    except (SettingsError, ConfigurationError) as e:
        print_rich_or_typer(f"[red]Settings error: {e}[/red]", stderr=True)
        raise typer.Exit(2) from e


def get_cli_defaults(command: str) -> JSONObject:
    """Pass-through for tests to monkeypatch at flujo.cli.main level.

    Delegates to the real config manager function unless monkeypatched in tests.
    """
    return _get_cli_defaults(command)


# Compatibility functions for testing - re-export functions that tests expect to monkeypatch
# These maintain the testing interface while the actual implementations live elsewhere


async def run_default_pipeline(
    pipeline: "Pipeline[str, Checklist]", task: "Task"
) -> "Optional[Candidate]":
    """Compatibility function for testing - re-exports from recipes.factories."""
    return await _run_default_pipeline(pipeline, task)


def make_review_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, Checklist]":
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_review_agent(model)


def make_solution_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, str]":
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_solution_agent(model)


def make_validator_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, Checklist]":
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_validator_agent(model)


def get_reflection_agent(model: str | None = None) -> Any:
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _get_reflection_agent(model)


def make_default_pipeline(**kwargs: Any) -> "Pipeline[str, Checklist]":
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


def make_self_improvement_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, Any]":
    """Compatibility function for testing - re-exports from agents.recipes."""
    return _make_self_improvement_agent(model)


def load_settings() -> "FlujoSettings":
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
