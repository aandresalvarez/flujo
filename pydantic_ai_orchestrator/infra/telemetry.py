"""Telemetry and logging for pydantic-ai-orchestrator.""" 

import logfire
import platform
from .settings import settings

_initialized = False

def init():
    """
    Initialize Logfire telemetry for the orchestrator.
    Uses API key from settings, or runs in local mode if not set.
    Idempotent: safe to call multiple times.
    """
    global _initialized
    if _initialized:
        return
    logfire.configure(
        service_name="pydantic_ai_orchestrator",
        api_key=settings.logfire_api_key.get_secret_value() if settings.logfire_api_key else None,
        attributes={"host": platform.node()},
    )
    _initialized = True 