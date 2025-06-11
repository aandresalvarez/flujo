"""
Pydantic AI Orchestrator package init.
"""
from importlib.metadata import version
from .application.orchestrator import Orchestrator
from .infra.settings import settings
from .infra.telemetry import init as _init_telemetry

__version__ = version("pydantic_ai_orchestrator")
__all__ = ["Orchestrator", "settings"]

_init_telemetry() 