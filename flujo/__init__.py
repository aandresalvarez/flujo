"""
Flujo package init.
"""

try:
    from importlib.metadata import version

    __version__ = version("flujo")
except Exception:
    __version__ = "0.0.0"
from .application.flujo_engine import Flujo
from .recipes import Default, AgenticLoop
from .infra.settings import settings
from .infra.telemetry import init_telemetry

from .domain.models import Task, Candidate, Checklist, ChecklistItem
from .domain import (
    Step,
    step,
    Pipeline,
    StepConfig,
    PluginOutcome,
    ValidationPlugin,
    AppResources,
)
from .domain.types import HookCallable
from .domain.events import HookPayload
from .domain.backends import ExecutionBackend, StepExecutionRequest
from .infra.backends import LocalBackend
from .tracing import ConsoleTracer
from .application.eval_adapter import run_pipeline_async
from .application.self_improvement import evaluate_and_improve, SelfImprovementAgent
from .domain.models import PipelineResult, StepResult, UsageLimits
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

from .exceptions import (
    OrchestratorError,
    ConfigurationError,
    SettingsError,
    UsageLimitExceededError,
    PipelineAbortSignal,
)

__all__ = [
    "Flujo",
    "Default",
    "AgenticLoop",
    "Task",
    "Candidate",
    "Checklist",
    "ChecklistItem",
    "Step",
    "step",
    "Pipeline",
    "StepConfig",
    "AppResources",
    "PluginOutcome",
    "ValidationPlugin",
    "run_pipeline_async",
    "evaluate_and_improve",
    "SelfImprovementAgent",
    "PipelineResult",
    "StepResult",
    "HookCallable",
    "HookPayload",
    "PipelineAbortSignal",
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
    "UsageLimitExceededError",
    "StubAgent",
    "DummyPlugin",
    "SQLSyntaxValidator",
    "UsageLimits",
    "ExecutionBackend",
    "StepExecutionRequest",
    "LocalBackend",
    "ConsoleTracer",
]
