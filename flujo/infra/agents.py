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


# Import the agent factory from the new dedicated module
from ..agents.factory import make_agent, _unwrap_type_adapter


class AsyncAgentWrapper(Generic[AgentInT, AgentOutT], AsyncAgentProtocol[AgentInT, AgentOutT]):
    """
    Wraps a pydantic_ai.Agent to provide an asynchronous interface
    with retry and timeout capabilities.
    """

    def __init__(
        self,
        agent: Agent[Any, AgentOutT],
        max_retries: int = 3,
        timeout: int | None = None,
        model_name: str | None = None,
        processors: Optional[AgentProcessors] = None,
        auto_repair: bool = True,
    ) -> None:
        if not isinstance(max_retries, int):
            raise TypeError(f"max_retries must be an integer, got {type(max_retries).__name__}.")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if timeout is not None:
            if not isinstance(timeout, int):
                raise TypeError(
                    f"timeout must be an integer or None, got {type(timeout).__name__}."
                )
            if timeout <= 0:
                raise ValueError("timeout must be a positive integer if specified.")
        self._agent = agent
        self._max_retries = max_retries
        from flujo.infra.settings import settings as current_settings

        self._timeout_seconds: int | None = (
            timeout if timeout is not None else current_settings.agent_timeout
        )
        self._model_name: str | None = model_name or getattr(agent, "model", "unknown_model")
        # Use centralized model ID extraction for consistency
        from ..utils.model_utils import extract_model_id

        self.model_id: str | None = model_name or extract_model_id(agent, "AsyncAgentWrapper")
        self.processors: AgentProcessors = processors or AgentProcessors()
        self.auto_repair = auto_repair
        self.target_output_type = getattr(agent, "output_type", Any)

    def _call_agent_with_dynamic_args(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying agent with arbitrary arguments."""

        # Check if the underlying agent accepts context parameters
        from flujo.application.core.context_manager import _accepts_param

        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in ["context", "pipeline_context"]:
                # Only pass context if the underlying agent accepts it
                if _accepts_param(self._agent.run, "context"):
                    filtered_kwargs["context"] = v
            else:
                filtered_kwargs[k] = v

        return self._agent.run(*args, **filtered_kwargs)

    async def _run_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent with retry, timeout and processor support."""

        # Get context from kwargs (supports both 'context' and legacy 'pipeline_context')
        context_obj = kwargs.get("context") or kwargs.get("pipeline_context")

        processed_args = list(args)
        if self.processors.prompt_processors and processed_args:
            prompt_data = processed_args[0]
            for proc in self.processors.prompt_processors:
                prompt_data = await proc.process(prompt_data, context_obj)
            processed_args[0] = prompt_data

        # Compatibility shim: pydantic-ai expects serializable dicts for its
        # internal function-calling message generation, not Pydantic model
        # instances. We automatically serialize any BaseModel inputs here to
        # ensure compatibility.
        processed_args = [
            arg.model_dump() if isinstance(arg, PydanticBaseModel) else arg
            for arg in processed_args
        ]

        # FR-35.2: Filter kwargs before processing to avoid passing unwanted parameters
        # This is the core fix for FSD-11 - only pass context if the underlying agent accepts it
        from flujo.application.core.context_manager import _accepts_param

        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in ["context", "pipeline_context"]:
                # Only pass context if the underlying agent's run method accepts it
                accepts_context = _accepts_param(self._agent.run, "context")
                if accepts_context:
                    filtered_kwargs[key] = value
                # Note: We don't pass context to the underlying agent if it doesn't accept it
                # This prevents the TypeError: run() got an unexpected keyword argument 'context'
            else:
                filtered_kwargs[key] = value

        processed_kwargs = {
            key: value.model_dump() if isinstance(value, PydanticBaseModel) else value
            for key, value in filtered_kwargs.items()
        }

        retryer = AsyncRetrying(
            reraise=False,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(max=60),
        )

        try:
            async for attempt in retryer:
                with attempt:
                    raw_agent_response = await asyncio.wait_for(
                        self._call_agent_with_dynamic_args(
                            *processed_args,
                            **processed_kwargs,
                        ),
                        timeout=self._timeout_seconds,
                    )
                    logfire.info(f"Agent '{self._model_name}' raw response: {raw_agent_response}")

                    if isinstance(raw_agent_response, str) and raw_agent_response.startswith(
                        "Agent failed after"
                    ):
                        raise OrchestratorRetryError(raw_agent_response)

                    # Store the original AgentRunResult for usage tracking
                    agent_usage_info = None
                    if hasattr(raw_agent_response, "usage"):
                        agent_usage_info = raw_agent_response.usage()

                    # Get the actual output content to be processed
                    unpacked_output = getattr(raw_agent_response, "output", raw_agent_response)

                    if self.processors.output_processors:
                        processed = unpacked_output
                        for proc in self.processors.output_processors:
                            processed = await proc.process(processed, context_obj)
                        unpacked_output = processed

                    # Return a tuple with both the processed output and usage info
                    # This ensures the usage information is preserved even after processing
                    if agent_usage_info is not None:
                        # Create a wrapper that preserves both the processed output and usage info
                        # This is a clean abstraction that maintains the contract expected by cost tracking
                        # while allowing output processors to work on the actual content
                        class ProcessedOutputWithUsage:
                            """
                            Wrapper that preserves both processed output and usage information.

                            This ensures that cost tracking can extract usage metrics even after
                            output processors have modified the content. The wrapper maintains
                            the same interface as the original AgentRunResult for usage extraction.
                            """

                            def __init__(self, output: Any, usage_info: Any) -> None:
                                self.output = output
                                self._usage_info = usage_info

                            def usage(self) -> Any:
                                """Return the original usage information from the agent response."""
                                return self._usage_info

                        return ProcessedOutputWithUsage(unpacked_output, agent_usage_info)
                    else:
                        return unpacked_output
        except RetryError as e:
            last_exc = e.last_attempt.exception()
            if isinstance(last_exc, (ValidationError, ModelRetry)) and self.auto_repair:
                logfire.warn(
                    f"Agent validation failed. Initiating automated repair. Error: {last_exc}"
                )
                raw_output = get_raw_output_from_exception(last_exc)
                try:
                    cleaner = DeterministicRepairProcessor()
                    cleaned = await cleaner.process(raw_output)
                    validated = TypeAdapter(
                        _unwrap_type_adapter(self.target_output_type)
                    ).validate_json(cleaned)
                    logfire.info("Deterministic repair successful.")
                    return validated
                except (ValidationError, ValueError, TypeError):
                    logfire.warn("Deterministic repair failed. Escalating to LLM repair.")
                try:
                    schema = TypeAdapter(
                        _unwrap_type_adapter(self.target_output_type)
                    ).json_schema()
                    prompt_data = {
                        "json_schema": json.dumps(safe_serialize(schema), ensure_ascii=False),
                        "original_prompt": str(args[0]) if args else "",
                        "failed_output": raw_output,
                        "validation_error": str(last_exc),
                    }
                    prompt = _format_repair_prompt(prompt_data)
                    repaired_str = await get_repair_agent().run(prompt)
                    try:
                        # First, try to parse the repair agent's response as JSON
                        repair_response = json.loads(repaired_str)

                        # Check if the repair agent explicitly signaled it cannot fix the output
                        if (
                            isinstance(repair_response, dict)
                            and repair_response.get("repair_error") is True
                        ):
                            reasoning = repair_response.get("reasoning", "No reasoning provided")
                            logfire.warn(f"Repair agent cannot fix output: {reasoning}")
                            raise OrchestratorError(f"Repair agent cannot fix output: {reasoning}")

                        # If not a repair error, validate against the target type
                        validated = TypeAdapter(
                            _unwrap_type_adapter(self.target_output_type)
                        ).validate_python(repair_response)
                        logfire.info("LLM repair successful.")
                        return validated
                    except json.JSONDecodeError as decode_exc:
                        logfire.error(
                            f"LLM repair failed: Invalid JSON returned by repair agent: {decode_exc}\nRaw output: {repaired_str}"
                        )
                        raise OrchestratorError(
                            f"Agent validation failed: repair agent returned invalid JSON: {decode_exc}\nRaw output: {repaired_str}"
                        )
                    except (ValidationError, ValueError, TypeError) as repair_exc:
                        logfire.warn(f"LLM repair failed: {repair_exc}\nRaw output: {repaired_str}")
                        raise OrchestratorError(
                            f"Agent validation failed: schema validation error: {repair_exc}\nRaw output: {repaired_str}"
                        )
                except Exception as repair_agent_exc:
                    logfire.warn(f"Repair agent failed: {repair_agent_exc}")
                    # Raise OrchestratorError for repair agent failures
                    raise OrchestratorError(f"Repair agent execution failed: {repair_agent_exc}")
            else:
                # FR-36: Enhanced error reporting with actual error type and message
                error_type = type(last_exc).__name__
                error_message = str(last_exc)
                logfire.error(
                    f"Agent '{self._model_name}' failed after {self._max_retries} attempts. Last error: {error_type}({error_message})"
                )
                # For timeout and retry scenarios, raise OrchestratorRetryError
                if isinstance(last_exc, (TimeoutError, asyncio.TimeoutError)):
                    raise OrchestratorRetryError(
                        f"Agent timed out after {self._max_retries} attempts"
                    )
                else:
                    raise OrchestratorRetryError(
                        f"Agent failed after {self._max_retries} attempts. Last error: {error_type}({error_message})"
                    )
        except Exception as e:
            # FR-36: Enhanced error reporting for non-retry errors
            error_type = type(e).__name__
            error_message = str(e)
            logfire.error(
                f"Agent '{self._model_name}' execution failed: {error_type}({error_message})"
            )
            # For timeout scenarios, raise OrchestratorRetryError
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                raise OrchestratorRetryError(f"Agent timed out: {error_message}")
            else:
                raise OrchestratorError(
                    f"Agent '{self._model_name}' execution failed: {error_type}({error_message})"
                )

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._run_with_retry(*args, **kwargs)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        return await self.run_async(*args, **kwargs)


def make_agent_async(
    model: str,
    system_prompt: str,
    output_type: Type[Any],
    max_retries: int = 3,
    timeout: int | None = None,
    processors: Optional[AgentProcessors] = None,
    auto_repair: bool = True,
    **kwargs: Any,
) -> AsyncAgentWrapper[Any, Any]:
    """
    Creates a pydantic_ai.Agent and returns an AsyncAgentWrapper exposing .run_async.

    Parameters
    ----------
    model : str
        The model identifier (e.g., "openai:gpt-4o")
    system_prompt : str
        The system prompt for the agent
    output_type : Type[Any]
        The expected output type
    max_retries : int, optional
        Maximum number of retries for failed calls
    timeout : int, optional
        Timeout in seconds for agent calls
    processors : Optional[AgentProcessors], optional
        Custom processors for the agent
    auto_repair : bool, optional
        Whether to enable automatic repair of failed outputs
    **kwargs : Any
        Additional arguments to pass to the underlying pydantic_ai.Agent
        (e.g., temperature, model_settings, max_tokens, etc.)
    """
    # Check if this is an image generation model
    is_image_model = _is_image_generation_model(model)

    agent, final_processors = make_agent(
        model,
        system_prompt,
        output_type,
        processors=processors,
        **kwargs,
    )

    # If this is an image model, attach the image cost post-processor
    if is_image_model:
        _attach_image_cost_post_processor(agent, model)

    return AsyncAgentWrapper(
        agent,
        max_retries=max_retries,
        timeout=timeout,
        model_name=model,
        processors=final_processors,
        auto_repair=auto_repair,
    )


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


def make_repair_agent(model: str | None = None) -> AsyncAgentWrapper[Any, str]:
    """Create the internal JSON repair agent."""
    model_name = model or settings.default_repair_model
    return make_agent_async(model_name, REPAIR_SYS, str, auto_repair=False)


_repair_agent: AsyncAgentWrapper[Any, str] | None = None


def get_repair_agent() -> AsyncAgentWrapper[Any, str]:
    """Lazily create the internal repair agent."""
    global _repair_agent
    if _repair_agent is None:
        _repair_agent = make_repair_agent()
    return _repair_agent


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
