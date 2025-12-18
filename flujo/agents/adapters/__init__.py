"""Agent adapters for different vendor backends.

This module provides adapters that convert vendor-specific agent responses
to Flujo's vendor-agnostic FlujoAgentResult interface.
"""

from __future__ import annotations

from .pydantic_ai_adapter import PydanticAIAdapter, PydanticAIUsageAdapter

__all__ = [
    "PydanticAIAdapter",
    "PydanticAIUsageAdapter",
]
