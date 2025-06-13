"""
Pydantic AI Orchestrator package init.
"""
try:
    from importlib.metadata import version
    __version__ = version("pydantic_ai_orchestrator")
except Exception:
    __version__ = "0.0.0"
from .application.orchestrator import Orchestrator
from .infra.settings import settings
from .infra.telemetry import init_telemetry

from .domain.models import Task, Candidate, Checklist, ChecklistItem
from .domain import Step, Pipeline, StepConfig, PluginOutcome, ValidationPlugin

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
]
