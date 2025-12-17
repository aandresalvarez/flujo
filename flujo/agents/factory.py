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

from .agent_like import AgentLike
from ..domain.processors import AgentProcessors
from ..exceptions import ConfigurationError
from ..utils.model_utils import extract_provider_and_model


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


class _LocalMockAgent:
    def __init__(self, model: str, output_type: Any) -> None:
        self.model: object = model
        self.output_type: object = output_type
        self.target_output_type: object = output_type

    async def run(self, *args: Any, **kwargs: Any) -> object:
        if args:
            return args[0]
        if "data" in kwargs:
            return kwargs["data"]
        return None

    async def run_async(self, *args: Any, **kwargs: Any) -> object:
        return await self.run(*args, **kwargs)


def make_agent(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    tools: list[Any] | None = None,
    processors: Optional[AgentProcessors] = None,
    **kwargs: Any,
) -> tuple[AgentLike, AgentProcessors]:
    """Creates a pydantic_ai.Agent, injecting the correct API key and returns it with processors."""
    provider_name = model.split(":")[0].lower()
    base_model = model.split(":", 1)[1].lower() if ":" in model else ""
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

    # Local stub model: allow offline/deterministic agents for tests and CLI fixtures.
    # This returns a minimal agent that simply echoes the first positional argument,
    # preserving type hints via output_type to satisfy validation paths without
    # requiring a real provider or network access.
    if provider_name == "local" and base_model == "mock":
        return _LocalMockAgent(model, actual_type), final_processors

    try:
        # Specialized handling for OpenAI GPT-5 and other reasoning models using Responses API
        provider, base_model = extract_provider_and_model(model)
        base_model_lc = (base_model or "").lower()

        # Detect GPT-5 family (gpt-5 and gpt-5-mini) and allow future o-series by prefix
        is_openai = (provider_name == "openai") or (provider == "openai")
        is_gpt5_family = base_model_lc.startswith("gpt-5")

        if is_openai and is_gpt5_family:
            # Import lazily to avoid module import errors when GPT-5 isn't used
            try:
                from pydantic_ai.models.openai import (
                    OpenAIResponsesModel,
                    OpenAIResponsesModelSettings,
                )
            except Exception as e:  # pragma: no cover - defensive
                raise ConfigurationError(
                    "GPT-5 requires pydantic-ai Responses model support (>=0.7). "
                    f"Import failed: {e}"
                )
            # Convert model_settings dict (if provided) into OpenAIResponsesModelSettings
            ms = kwargs.pop("model_settings", None)
            settings_obj: Any | None = None
            if isinstance(ms, dict):
                try:
                    settings_obj = OpenAIResponsesModelSettings(**ms)  # type: ignore[typeddict-item]
                except Exception as e:  # pragma: no cover - defensive
                    raise ConfigurationError(f"Invalid model_settings for GPT-5: {ms} ({e})")
            elif ms is not None:
                # If user already provided a typed settings object, accept it
                settings_obj = ms

            # Build Responses wrapper for GPT-5
            responses_model = OpenAIResponsesModel(base_model)

            agent = Agent(
                model=responses_model,
                system_prompt=system_prompt,
                output_type=actual_type,
                tools=tools or [],
                # Pass typed settings (Pydantic AI will ignore None)
                model_settings=settings_obj,
                **kwargs,
            )
        else:
            # Default path: rely on pydantic-ai to resolve profile from string
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
