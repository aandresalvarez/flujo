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
from .domain.models import Task

__all__ = ["Orchestrator", "settings", "Task"]

init_telemetry() 