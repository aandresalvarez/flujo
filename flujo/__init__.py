"""
Flujo package init.
"""

try:
    from importlib.metadata import version

    __version__ = version("flujo")
except Exception:
    __version__ = "0.0.0"
from .application.orchestrator import Orchestrator
from .infra.settings import settings
from .infra.telemetry import init_telemetry

from .domain.models import Task, Candidate, Checklist, ChecklistItem
from .domain import (
    Step,
    Pipeline,
    StepConfig,
    PluginOutcome,
    ValidationPlugin,
)
from .application.pipeline_runner import PipelineRunner
from .application.eval_adapter import run_pipeline_async
from .application.self_improvement import evaluate_and_improve, SelfImprovementAgent
from .domain.models import PipelineResult, StepResult
from .testing.utils import StubAgent, DummyPlugin
from .plugins.sql_validator import SQLSyntaxValidator

from .infra.agents import (
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
    get_reflection_agent,
    make_agent_async,
)

from .exceptions import OrchestratorError, ConfigurationError, SettingsError

__all__ = [
    "Orchestrator",
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "Step",
    "Pipeline",
    "StepConfig",
    "PluginOutcome",
    "ValidationPlugin",
    "PipelineRunner",
    "run_pipeline_async",
    "evaluate_and_improve",
    "SelfImprovementAgent",
    "PipelineResult",
    "StepResult",
    "settings",
    "init_telemetry",
    "review_agent",
    "solution_agent",
    "validator_agent",
    "reflection_agent",
    "get_reflection_agent",
    "make_agent_async",
    "OrchestratorError",
    "ConfigurationError",
    "SettingsError",
    "StubAgent",
    "DummyPlugin",
    "SQLSyntaxValidator",
]
