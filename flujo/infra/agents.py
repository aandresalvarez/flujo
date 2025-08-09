"""
Agent factory utilities and wrapper classes.

This module provides factory functions for creating agents and wrapper classes
for async execution. It focuses on agent creation and resilience wrapping,
while system prompts are now in the flujo.prompts module.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any, Optional, Type, Generic, get_origin

from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry
from pydantic import BaseModel as PydanticBaseModel, TypeAdapter
import os
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from ..domain.agent_protocol import AsyncAgentProtocol, AgentInT, AgentOutT
from ..domain.models import Checklist, ImprovementReport
from ..domain.processors import AgentProcessors
from ..exceptions import OrchestratorError, OrchestratorRetryError, ConfigurationError
from ..utils.serialization import safe_serialize
from ..processors.repair import DeterministicRepairProcessor
from .settings import settings
from .telemetry import logfire

# Import prompts from the new prompts module
from ..prompts import (
    REVIEW_SYS,
    SOLUTION_SYS,
    VALIDATE_SYS,
    REFLECT_SYS,
    SELF_IMPROVE_SYS,
    REPAIR_SYS,
    _format_repair_prompt,
)


def get_raw_output_from_exception(exc: Exception) -> str:
    """Best-effort extraction of raw output from validation-related exceptions."""
    if hasattr(exc, "message"):
        msg = getattr(exc, "message")
        if isinstance(msg, str):
            return msg
    if exc.args:
        first = exc.args[0]
        if isinstance(first, str):
            return first
    return str(exc)


# Import the agent factory and wrapper from the new dedicated modules
from ..agents.factory import make_agent, _unwrap_type_adapter
from ..agents.wrapper import AsyncAgentWrapper, make_agent_async

# Alias for backward compatibility with tests that monkeypatch this module
make_agent_async = make_agent_async


# The AsyncAgentWrapper class has been moved to flujo.agents.wrapper
# This import maintains backward compatibility



# Model registry for image generation models
IMAGE_GENERATION_MODEL_PATTERNS = {
    "openai": [
        "dall-e",
        "dall-e-2",
        "dall-e-3",
    ],
    "midjourney": [
        "midjourney",
        "mj",
    ],
    "stability": [
        "stable-diffusion",
        "sd",
    ],
    "stable-diffusion": [
        "stable-diffusion",
        "sd",
        "xl",
    ],
    "google": [
        "imagen",
        "imagen-2",
    ],
    "anthropic": [
        "claude-3-haiku-image",
        "claude-3-sonnet-image",
        "claude-3-opus-image",
    ],
    "meta": [
        "emma",
        "emma-2",
    ],
}


def _is_image_generation_model(model: str) -> bool:
    """
    Check if the model is an image generation model.

    This function examines the model identifier to determine if it's an image
    generation model using a configuration-based approach for better maintainability
    and extensibility.

    Parameters
    ----------
    model : str
        The model identifier (e.g., "openai:dall-e-3")

    Returns
    -------
    bool
        True if the model is an image generation model
    """
    # Handle edge cases
    if not model:
        return False

    # Extract the provider and model name from the provider:model format
    if ":" in model:
        provider = model.split(":", 1)[0].lower()
        model_name = model.split(":", 1)[1].lower()

        # Handle case where model name is empty (e.g., "openai:")
        if not model_name:
            return False
    else:
        provider = ""
        model_name = model.lower()

    # Check against the model registry
    for provider_patterns in IMAGE_GENERATION_MODEL_PATTERNS.values():
        for pattern in provider_patterns:
            if pattern in model_name:
                return True

    # Only check provider if it's specifically an image generation provider
    # (not just any provider that has image models)
    image_only_providers = {"midjourney", "stability", "stable-diffusion"}
    if provider in image_only_providers:
        return True

    return False


def _attach_image_cost_post_processor(agent: Any, model: str) -> None:
    """
    Attach the image cost post-processor to an agent.

    Parameters
    ----------
    agent : Any
        The pydantic-ai Agent to attach the post-processor to
    model : str
        The model identifier for loading pricing configuration
    """
    from ..cost import _image_cost_post_processor
    from ..infra.config import get_provider_pricing
    from ..utils.model_utils import extract_provider_and_model
    from ..infra import telemetry

    try:
        # Extract provider and model name
        provider, model_name = extract_provider_and_model(model)

        if provider is None:
            telemetry.logfire.warning(
                f"Could not determine provider for model '{model}'. "
                f"Image cost post-processor will not be attached."
            )
            return

        # Get pricing configuration
        pricing = get_provider_pricing(provider, model_name)

        if pricing is None:
            telemetry.logfire.warning(
                f"No pricing configuration found for '{provider}:{model_name}'. "
                f"Image cost post-processor will not be attached."
            )
            return

        # Extract image pricing data from the pricing object
        pricing_data = {}
        for field_name, field_value in pricing.model_dump().items():
            if field_name.startswith("price_per_image_") and field_value is not None:
                pricing_data[field_name] = field_value

        if not pricing_data:
            telemetry.logfire.warning(
                f"No image pricing found for '{provider}:{model_name}'. "
                f"Image cost post-processor will not be attached."
            )
            return

        # Create a partial function with the pricing data bound
        from functools import partial

        post_processor = partial(_image_cost_post_processor, pricing_data=pricing_data)

        # Attach the post-processor to the agent
        if not hasattr(agent, "post_processors"):
            agent.post_processors = []

        agent.post_processors.append(post_processor)

        telemetry.logfire.info(
            f"Attached image cost post-processor to '{model}' "
            f"with pricing keys: {list(pricing_data.keys())}"
        )

    except Exception as e:
        telemetry.logfire.warning(f"Failed to attach image cost post-processor to '{model}': {e}")


class NoOpReflectionAgent(AsyncAgentProtocol[Any, str]):
    """A stub agent that does nothing, used when reflection is disabled."""

    async def run(self, data: Any | None = None, **kwargs: Any) -> str:
        return ""

    async def run_async(self, data: Any | None = None, **kwargs: Any) -> str:
        return ""


class NoOpChecklistAgent(AsyncAgentProtocol[Any, Checklist]):
    """A stub agent that returns an empty Checklist, used as a fallback for checklist agents."""

    async def run(self, data: Any | None = None, **kwargs: Any) -> Checklist:
        return Checklist(items=[])

    async def run_async(self, data: Any | None = None, **kwargs: Any) -> Checklist:
        return Checklist(items=[])


def get_reflection_agent(
    model: str | None = None,
) -> AsyncAgentProtocol[Any, Any] | NoOpReflectionAgent:
    """Returns a new instance of the reflection agent, or a no-op if disabled."""
    if not settings.reflection_enabled:
        return NoOpReflectionAgent()
    try:
        model_name = model or settings.default_reflection_model
        agent = make_agent_async(model_name, REFLECT_SYS, str)
        logfire.info("Reflection agent created successfully.")
        return agent
    except Exception as e:
        logfire.error(f"Failed to create reflection agent: {e}")
        return NoOpReflectionAgent()


def make_self_improvement_agent(
    model: str | None = None,
) -> AsyncAgentWrapper[Any, ImprovementReport]:
    """Create the SelfImprovementAgent."""
    model_name = model or settings.default_self_improvement_model
    return make_agent_async(model_name, SELF_IMPROVE_SYS, ImprovementReport)


# Repair agent functions have been moved to flujo.agents.wrapper
from ..agents.wrapper import get_repair_agent


def make_repair_agent(model: str | None = None) -> AsyncAgentWrapper[Any, str]:
    """Create the internal JSON repair agent."""
    model_name = model or settings.default_repair_model
    return make_agent_async(model_name, REPAIR_SYS, str, auto_repair=False)


# Factory functions for creating default agents
def make_review_agent(model: str | None = None) -> AsyncAgentWrapper[Any, Checklist]:
    """Create a review agent with default settings."""
    model_name = model or settings.default_review_model
    return make_agent_async(model_name, REVIEW_SYS, Checklist)


def make_solution_agent(model: str | None = None) -> AsyncAgentWrapper[Any, str]:
    """Create a solution agent with default settings."""
    model_name = model or settings.default_solution_model
    return make_agent_async(model_name, SOLUTION_SYS, str)


def make_validator_agent(model: str | None = None) -> AsyncAgentWrapper[Any, Checklist]:
    """Create a validator agent with default settings."""
    model_name = model or settings.default_validator_model
    return make_agent_async(model_name, VALIDATE_SYS, Checklist)


class LoggingReviewAgent(AsyncAgentProtocol[Any, Any]):
    """Wrapper for review agent that adds logging."""

    def __init__(self, agent: AsyncAgentProtocol[Any, Any]) -> None:
        self.agent = agent

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_inner(self.agent.run, *args, **kwargs)

    async def _run_async(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.agent, "run_async") and callable(getattr(self.agent, "run_async")):
            return await self._run_inner(self.agent.run_async, *args, **kwargs)
        else:
            return await self.run(*args, **kwargs)

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_async(*args, **kwargs)

    async def _run_inner(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await method(*args, **kwargs)
            logfire.info(f"Review agent result: {result}")
            return result
        except Exception as e:
            logfire.error(f"Review agent API error: {e}")
            raise


# Explicit exports
__all__ = [
    "make_agent",
    "make_agent_async",
    "AsyncAgentWrapper",
    "NoOpReflectionAgent",
    "NoOpChecklistAgent",
    "get_reflection_agent",
    "make_self_improvement_agent",
    "make_repair_agent",
    "get_repair_agent",
    "make_review_agent",
    "make_solution_agent",
    "make_validator_agent",
    "LoggingReviewAgent",
    "Agent",
    "AsyncAgentProtocol",
    "AgentInT",
    "AgentOutT",
]


# Deprecation warnings for removed global agents
def _deprecated_agent(name: str) -> None:
    """Create a deprecation warning for removed global agents."""
    warnings.warn(
        f"The global {name} instance has been removed. "
        f"Use make_{name}_agent() to create a new instance instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    raise AttributeError(
        f"Global {name} instance has been removed. Use make_{name}_agent() instead."
    )


# Define deprecated global agents that raise helpful errors
def __getattr__(name: str) -> Any:
    """Handle access to removed global agent instances."""
    if name == "review_agent":
        _deprecated_agent("review")
    elif name == "solution_agent":
        _deprecated_agent("solution")
    elif name == "validator_agent":
        _deprecated_agent("validator")
    elif name == "reflection_agent":
        warnings.warn(
            "The global reflection_agent instance has been removed. "
            "Use get_reflection_agent() to create a new instance instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(
            "Global reflection_agent instance has been removed. Use get_reflection_agent() instead."
        )
    elif name == "self_improvement_agent":
        warnings.warn(
            "The global self_improvement_agent instance has been removed. "
            "Use make_self_improvement_agent() to create a new instance instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(
            "Global self_improvement_agent instance has been removed. Use make_self_improvement_agent() instead."
        )
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
