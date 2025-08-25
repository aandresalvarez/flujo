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
        # Defer hard API key requirement to runtime. This allows tests to
        # construct agents and monkeypatch their run methods without needing
        # real credentials. If a real call is made without a key, the provider
        # will raise at execution time.
        if current_settings.openai_api_key:
            os.environ.setdefault(
                "OPENAI_API_KEY", current_settings.openai_api_key.get_secret_value()
            )
        else:
            # Provide a benign placeholder for libraries that require an env var
            os.environ.setdefault("OPENAI_API_KEY", "test")
    elif provider_name in {"google-gla", "gemini"}:
        if current_settings.google_api_key:
            os.environ.setdefault(
                "GOOGLE_API_KEY", current_settings.google_api_key.get_secret_value()
            )
        else:
            os.environ.setdefault("GOOGLE_API_KEY", "test")
    elif provider_name == "anthropic":
        # For Anthropic, require a real API key to be configured to prevent
        # accidental runtime calls with placeholders. Honor either settings
        # or existing environment variables.
        configured_key = (
            current_settings.anthropic_api_key.get_secret_value()
            if current_settings.anthropic_api_key
            else None
        )
        env_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ORCH_ANTHROPIC_API_KEY")
        effective_key = configured_key or env_key
        if not effective_key:
            # Enforce fail-fast configuration error for missing Anthropic key
            raise ConfigurationError(
                "Anthropic API key is required (set settings.anthropic_api_key or ANTHROPIC_API_KEY)."
            )
        # Ensure downstream libraries see the key
        os.environ.setdefault("ANTHROPIC_API_KEY", effective_key)

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
