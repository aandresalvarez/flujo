"""
Infrastructure components for flujo.
"""

from .settings import settings
from .telemetry import init_telemetry
from .backends import LocalBackend

# Agent functions have been moved to flujo.agents package
from .config_manager import (
    load_settings,
    get_cli_defaults,
    get_state_uri,
)

__all__ = [
    "settings",
    "init_telemetry",
    "LocalBackend",
    # Configuration management functions
    "load_settings",
    "get_cli_defaults",
    "get_state_uri",
]
