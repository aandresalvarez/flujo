"""
High-level agent recipes and factory functions.

This module contains application-specific agent creation functions and
no-op agent implementations. These recipes build on the base factory
and wrapper functionality to provide domain-specific agents.

Extracted from flujo.infra.agents as part of FSD-005.3 to isolate
application-specific agent implementations from general framework machinery.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..domain.agent_protocol import AsyncAgentProtocol
from ..domain.models import Checklist, ImprovementReport
from ..infra.telemetry import logfire

# Import the wrapper and factory functionality (avoiding circular imports)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .wrapper import AsyncAgentWrapper


# Import prompts from the prompts module
from ..prompts import (
    REVIEW_SYS,
    SOLUTION_SYS,
    VALIDATE_SYS,
    REFLECT_SYS,
    SELF_IMPROVE_SYS,
)

# Model pattern configuration for image generation detection
IMAGE_GENERATION_MODEL_PATTERNS = {
    "dall-e": ["dall-e", "dalle"],
    "midjourney": ["midjourney", "mj"],
    "stable-diffusion": ["stable-diffusion", "sdxl", "sd3"],
    "flux": ["flux"],
    "imagen": ["imagen"],
    "ideogram": ["ideogram"],
    "luna": ["luna"],
    "emma": ["emma", "emma-2"],
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
    from ..infra.settings import settings

    if not settings.reflection_enabled:
        return NoOpReflectionAgent()
    try:
        # Import here to avoid circular import issues
        from .wrapper import make_agent_async

        model_name = model or settings.default_reflection_model
        agent = make_agent_async(model_name, REFLECT_SYS, str)
        logfire.info("Reflection agent created successfully.")
        return agent
    except Exception as e:
        logfire.error(f"Failed to create reflection agent: {e}")
        return NoOpReflectionAgent()


def make_self_improvement_agent(
    model: str | None = None,
) -> "AsyncAgentWrapper[Any, ImprovementReport]":
    """Create the SelfImprovementAgent."""
    from ..infra.settings import settings

    # Import here to avoid circular import issues
    from .wrapper import make_agent_async

    model_name = model or settings.default_self_improvement_model
    return make_agent_async(model_name, SELF_IMPROVE_SYS, ImprovementReport)


# Factory functions for creating default agents
def make_review_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, Checklist]":
    """Create a review agent with default settings."""
    from ..infra.settings import settings

    # Import here to avoid circular import issues
    from .wrapper import make_agent_async

    model_name = model or settings.default_review_model
    return make_agent_async(model_name, REVIEW_SYS, Checklist)


def make_solution_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, str]":
    """Create a solution agent with default settings."""
    from ..infra.settings import settings

    # Import here to avoid circular import issues
    from .wrapper import make_agent_async

    model_name = model or settings.default_solution_model
    return make_agent_async(model_name, SOLUTION_SYS, str)


def make_validator_agent(model: str | None = None) -> "AsyncAgentWrapper[Any, Checklist]":
    """Create a validator agent with default settings."""
    from ..infra.settings import settings

    # Import here to avoid circular import issues
    from .wrapper import make_agent_async

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
    deprecated_names = [
        "review_agent",
        "solution_agent",
        "validator_agent",
        "reflection_agent",
        "self_improvement_agent",
        "repair_agent",
    ]
    if name in deprecated_names:
        _deprecated_agent(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
