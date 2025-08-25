"""
Asynchronous agent wrapper utilities.

This module provides the AsyncAgentWrapper class which enhances agents with:
- Asynchronous execution capabilities
- Retry logic with exponential backoff
- Timeout handling
- Error handling and automatic repair
- Processor integration

Extracted from flujo.infra.agents as part of FSD-005.2 to follow the
Single Responsibility Principle and isolate agent enhancement concerns
from agent creation concerns.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Type, Generic

from pydantic import ValidationError, BaseModel as PydanticBaseModel, TypeAdapter
from pydantic_ai import Agent, ModelRetry
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from ..domain.agent_protocol import AsyncAgentProtocol, AgentInT, AgentOutT
from ..domain.processors import AgentProcessors
from ..exceptions import OrchestratorError, OrchestratorRetryError
from ..utils.serialization import safe_serialize
from .repair import DeterministicRepairProcessor
from ..infra.telemetry import logfire

# Import the agent factory from the new dedicated module
from .factory import make_agent, _unwrap_type_adapter

# Import prompts from the prompts module
from ..prompts import _format_repair_prompt
from ..utils import format_prompt


# Import from utils to avoid circular imports
# Import the module (not the symbol) so tests can monkeypatch it
from . import utils as agents_utils


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
                    # Preserve the exact raw response object for downstream tracing/persistence
                    # This is intentionally stored on the wrapper instance to avoid changing
                    # the value type returned to policies and processors.
                    try:
                        self._last_raw_response = raw_agent_response
                    except Exception:
                        # Never let tracing state break execution
                        pass
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
                            """Wrapper that preserves processed output and usage information."""

                            def __init__(self, output: Any, usage_info: Any) -> None:
                                self.output = output
                                self._usage_info = usage_info

                            def usage(self) -> Any:
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
                # Use module reference to allow monkeypatching in tests
                raw_output = agents_utils.get_raw_output_from_exception(last_exc)
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
                    # Import here to avoid circular imports and allow monkeypatching
                    from .repair import get_repair_agent as repair_get_repair_agent

                    repair_agent = repair_get_repair_agent()
                    # Extract the actual string output from the repair agent response
                    repair_response = await repair_agent.run(prompt)
                    # Handle case where repair agent returns ProcessedOutputWithUsage
                    if hasattr(repair_response, "output"):
                        repaired_str = repair_response.output
                    else:
                        repaired_str = repair_response
                    try:
                        # First, try to parse the repair agent's response as JSON
                        # Handle case where output is wrapped in markdown code blocks
                        json_str = repaired_str
                        if json_str.startswith("```json\n") and json_str.endswith("\n```"):
                            json_str = json_str[8:-4].strip()
                        elif json_str.startswith("```\n") and json_str.endswith("\n```"):
                            json_str = json_str[4:-4].strip()

                        repair_response = json.loads(json_str)

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


class TemplatedAsyncAgentWrapper(AsyncAgentWrapper[AgentInT, AgentOutT]):
    """
    Async wrapper that supports just-in-time system prompt rendering from a template
    using runtime context and previous step output.

    The wrapper temporarily overrides the underlying agent's system_prompt for a single
    run and restores it afterwards to keep agent instances stateless.
    """

    def __init__(
        self,
        agent: Agent[Any, AgentOutT],
        *,
        template_string: str,
        variables_spec: Optional[dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: int | None = None,
        model_name: str | None = None,
        processors: Optional[AgentProcessors] = None,
        auto_repair: bool = True,
    ) -> None:
        super().__init__(
            agent,
            max_retries=max_retries,
            timeout=timeout,
            model_name=model_name,
            processors=processors,
            auto_repair=auto_repair,
        )
        self.system_prompt_template: str = template_string
        self.prompt_variables: dict[str, Any] = variables_spec or {}
        self._prompt_lock = asyncio.Lock()

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        # Derive previous_step from args or kwargs
        previous_step = args[0] if args else kwargs.get("previous_step") or None
        context = kwargs.get("context") or kwargs.get("pipeline_context")

        if self.system_prompt_template:
            # Resolve variable specs: support static values or template strings
            resolved_vars: dict[str, Any] = {}
            for key, value_template in (self.prompt_variables or {}).items():
                if isinstance(value_template, str) and "{{" in value_template:
                    try:
                        # Provide steps/context proxies for richer resolution
                        from ..utils.template_vars import (
                            get_steps_map_from_context,
                            StepValueProxy,
                            TemplateContextProxy,
                        )

                        steps_map = get_steps_map_from_context(context)
                        steps_wrapped = {
                            k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                            for k, v in steps_map.items()
                        }
                        ctx_proxy = TemplateContextProxy(context, steps=steps_wrapped)
                        resolved_vars[key] = format_prompt(
                            value_template,
                            context=ctx_proxy,
                            previous_step=previous_step,
                            steps=steps_wrapped,
                        )
                    except Exception:
                        resolved_vars[key] = ""
                else:
                    resolved_vars[key] = value_template

            # Render final system prompt
            try:
                from ..utils.template_vars import (
                    get_steps_map_from_context,
                    StepValueProxy,
                    TemplateContextProxy,
                )

                steps_map = get_steps_map_from_context(context)
                steps_wrapped = {
                    k: v if isinstance(v, StepValueProxy) else StepValueProxy(v)
                    for k, v in steps_map.items()
                }
                ctx_proxy = TemplateContextProxy(context, steps=steps_wrapped)
                final_system_prompt = format_prompt(
                    self.system_prompt_template,
                    **resolved_vars,
                    context=ctx_proxy,
                    previous_step=previous_step,
                    steps=steps_wrapped,
                )
            except Exception:
                final_system_prompt = self.system_prompt_template

            # Temporarily override system prompt with concurrency protection
            async with self._prompt_lock:
                original_prompt = getattr(self._agent, "system_prompt", None)
                try:
                    setattr(self._agent, "system_prompt", final_system_prompt)
                    return await super().run_async(*args, **kwargs)
                finally:
                    try:
                        setattr(self._agent, "system_prompt", original_prompt)
                    except Exception:
                        pass
        # No template configured; behave like base class
        return await super().run_async(*args, **kwargs)


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
    from .recipes import _is_image_generation_model, _attach_image_cost_post_processor

    is_image_model = _is_image_generation_model(model)

    # Import make_agent via infra path to allow test monkeypatching
    try:
        from flujo.agents import make_agent as infra_make_agent

        agent, final_processors = infra_make_agent(
            model,
            system_prompt,
            output_type,
            processors=processors,
            **kwargs,
        )
    except ImportError:
        # Fallback to direct import
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


def make_templated_agent_async(
    model: str,
    template_string: str,
    variables_spec: Optional[dict[str, Any]],
    output_type: Type[Any],
    max_retries: int = 3,
    timeout: int | None = None,
    processors: Optional[AgentProcessors] = None,
    auto_repair: bool = True,
    **kwargs: Any,
) -> TemplatedAsyncAgentWrapper[Any, Any]:
    """
    Create an agent and wrap it with TemplatedAsyncAgentWrapper to enable
    just-in-time system prompt rendering.
    """
    # Create underlying agent with a placeholder prompt; it will be overridden at runtime
    try:
        from flujo.agents import make_agent as infra_make_agent

        agent, final_processors = infra_make_agent(
            model,
            system_prompt="",
            output_type=output_type,
            processors=processors,
            **kwargs,
        )
    except ImportError:
        agent, final_processors = make_agent(
            model,
            system_prompt="",
            output_type=output_type,
            processors=processors,
            **kwargs,
        )

    return TemplatedAsyncAgentWrapper(
        agent,
        template_string=template_string,
        variables_spec=variables_spec,
        max_retries=max_retries,
        timeout=timeout,
        model_name=model,
        processors=final_processors,
        auto_repair=auto_repair,
    )
