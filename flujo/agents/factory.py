"""
Agent factory utilities.

This module provides the core agent creation functionality, focusing on:
- Generic agent creation with proper API key management
- Type adapter unwrapping utilities

Extracted from flujo.infra.agents as part of FSD-005.1 to follow the
Single Responsibility Principle and isolate agent creation concerns.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type, get_origin

from pydantic import TypeAdapter
from pydantic_ai import Agent

from ..domain.processors import AgentProcessors
from ..exceptions import ConfigurationError


def _unwrap_type_adapter(output_type: Any) -> Any:
    """Return the real type, unwrapping TypeAdapter instances."""
    if isinstance(output_type, TypeAdapter):
        return getattr(output_type, "annotation", getattr(output_type, "_type", output_type))
    origin = get_origin(output_type)
    if origin is TypeAdapter:
        args = getattr(output_type, "__args__", None)
        if args:
            return args[0]
    return output_type


def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
    processors: Optional[AgentProcessors] = None,
    **kwargs: Any,
) -> tuple[Agent[Any, Any], AgentProcessors]:
    """Creates a pydantic_ai.Agent, injecting the correct API key and returns it with processors."""
    provider_name = model.split(":")[0].lower()
    from flujo.infra.settings import get_settings

    current_settings = get_settings()

    if provider_name == "openai":
        if not current_settings.openai_api_key:
            raise ConfigurationError(
                "To use OpenAI models, the OPENAI_API_KEY environment variable must be set."
            )
        os.environ.setdefault("OPENAI_API_KEY", current_settings.openai_api_key.get_secret_value())
    elif provider_name in {"google-gla", "gemini"}:
        if not current_settings.google_api_key:
            raise ConfigurationError(
                "To use Gemini models, the GOOGLE_API_KEY environment variable must be set."
            )
        os.environ.setdefault("GOOGLE_API_KEY", current_settings.google_api_key.get_secret_value())
    elif provider_name == "anthropic":
        if not current_settings.anthropic_api_key:
            raise ConfigurationError(
                "To use Anthropic models, the ANTHROPIC_API_KEY environment variable must be set."
            )
        os.environ.setdefault(
            "ANTHROPIC_API_KEY", current_settings.anthropic_api_key.get_secret_value()
        )

    final_processors = processors.model_copy(deep=True) if processors else AgentProcessors()

    actual_type = _unwrap_type_adapter(output_type)

    try:
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            output_type=actual_type,
            tools=tools or [],
            **kwargs,
        )
    except (ValueError, TypeError, RuntimeError) as e:  # pragma: no cover - defensive
        raise ConfigurationError(f"Failed to create pydantic-ai agent: {e}") from e

    return agent, final_processors
