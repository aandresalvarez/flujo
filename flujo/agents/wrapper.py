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
from pydantic_ai import ModelRetry


from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from ..domain.agent_protocol import AsyncAgentProtocol, AgentInT, AgentOutT
from ..domain.processors import AgentProcessors
from ..exceptions import AgentIOValidationError, ExecutionError, OrchestratorRetryError
from .repair import DeterministicRepairProcessor
from ..infra.telemetry import logfire
from ..tracing.manager import get_active_trace_manager
from ..utils.redact import summarize_and_redact_prompt

# Import the agent factory from the new dedicated module
from .factory import make_agent, _unwrap_type_adapter

# Import prompts from the prompts module
from ..prompts import _format_repair_prompt
from ..utils import format_prompt


# Import from utils to avoid circular imports
# Import the module (not the symbol) so tests can monkeypatch it
from . import utils as agents_utils
from .agent_like import AgentLike
from .adapters.pydantic_ai_adapter import PydanticAIAdapter
from ..domain.agent_result import FlujoAgentResult


# pydantic-ai exception mapping (internal implementation detail)
# These are mapped to Flujo exceptions before being raised externally
def _resolve_unexpected_model_behavior() -> type[Exception]:  # pragma: no cover - import shim
    """Resolve pydantic-ai UnexpectedModelBehavior exception type.

    This is an internal implementation detail used for exception mapping.
    External code should only see Flujo exceptions.
    """
    try:  # pydantic-ai >=0.7
        from pydantic_ai.exceptions import UnexpectedModelBehavior as umb_imported

        return umb_imported
    except ImportError:

        class FallbackUnexpectedModelBehavior(Exception):
            pass

        return FallbackUnexpectedModelBehavior


UnexpectedModelBehavior: type[Exception] = _resolve_unexpected_model_behavior()


class AsyncAgentWrapper(Generic[AgentInT, AgentOutT], AsyncAgentProtocol[AgentInT, AgentOutT]):
    """
    Wraps a pydantic_ai.Agent to provide an asynchronous interface
    with retry and timeout capabilities.
    """

    def __init__(
        self,
        agent: AgentLike,
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

        # Respect provided timeout or fall back to global agent_timeout.
        # Do not clamp in test mode — tests expect parity with settings.
        base_timeout: int = timeout if timeout is not None else current_settings.agent_timeout
        self._timeout_seconds = base_timeout
        self._model_name: str | None = model_name or getattr(agent, "model", "unknown_model")
        # Use centralized model ID extraction for consistency
        from ..utils.model_utils import extract_model_id

        self.model_id: str | None = model_name or extract_model_id(agent, "AsyncAgentWrapper")
        self.processors: AgentProcessors = processors or AgentProcessors()
        self.auto_repair = auto_repair
        self.target_output_type = getattr(agent, "output_type", Any)
        # Optional structured output configuration (best-effort, pydantic-ai centric)
        self._structured_output_config: dict[str, Any] | None = None
        # Create adapter to convert pydantic-ai responses to FlujoAgentResult
        self._adapter = PydanticAIAdapter(agent)

    def _call_agent_with_dynamic_args(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying agent with arbitrary arguments."""

        # Check if the underlying agent accepts context parameters
        from flujo.application.core.context.context_manager import _accepts_param

        filtered_kwargs: dict[str, Any] = {}
        # Context/pipeline_context
        for k, v in kwargs.items():
            if k in ["context", "pipeline_context"]:
                if _accepts_param(self._agent.run, "context"):
                    filtered_kwargs["context"] = v
        # Structured options pass-through: expand known keys if underlying agent doesn't accept 'options'
        options = kwargs.get("options")
        if isinstance(options, dict):
            if _accepts_param(self._agent.run, "options"):
                filtered_kwargs["options"] = options
            else:
                # Expand only keys accepted by the underlying agent
                for ok, ov in options.items():
                    if _accepts_param(self._agent.run, str(ok)):
                        filtered_kwargs[str(ok)] = ov
        # Pass any other explicitly recognized kwargs that the underlying agent accepts
        for k, v in kwargs.items():
            if k in ("context", "pipeline_context", "options"):
                continue
            acc = _accepts_param(self._agent.run, str(k))
            if acc is not False:
                filtered_kwargs[str(k)] = v

        # Always forward primary payload under the conventional 'data' kwarg when provided.
        # Many tests use mocks or duck-typed agents whose signatures are not introspectable;
        # in those cases _accepts_param may return False even though the agent supports **kwargs.
        if "data" in kwargs and "data" not in filtered_kwargs:
            filtered_kwargs["data"] = kwargs["data"]

        return self._agent.run(*args, **filtered_kwargs)

    async def _run_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent with retry, timeout and processor support.

        Returns:
            FlujoAgentResult: Vendor-agnostic result containing output and usage metrics.
            Note: Return type is Any for protocol compatibility, but this method
            always returns FlujoAgentResult at runtime.
        """

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
        from flujo.application.core.context.context_manager import _accepts_param

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

        # Attach structured output configuration when present (best-effort)
        if self._structured_output_config:
            try:
                # Prefer direct response_format param
                if "response_format" not in processed_kwargs:
                    processed_kwargs["response_format"] = self._structured_output_config
            except Exception:
                pass

        retryer = AsyncRetrying(
            reraise=False,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(max=60),
        )

        try:
            async for attempt in retryer:
                with attempt:
                    # Prepare trace helpers
                    tm = get_active_trace_manager()
                    try:
                        import os as _os

                        debug_prompts = _os.getenv("FLUJO_DEBUG_PROMPTS") == "1"
                        try:
                            preview_len_env = int(_os.getenv("FLUJO_TRACE_PREVIEW_LEN", "1000"))
                        except Exception:
                            preview_len_env = 1000
                    except Exception:
                        debug_prompts = False
                        preview_len_env = 1000

                    # Record agent.system and agent.input events before call
                    try:
                        if tm is not None:
                            sys_txt_raw = getattr(self._agent, "system_prompt", None)
                            try:
                                # Handle callable/system_prompt methods returning text
                                if callable(sys_txt_raw):
                                    sys_val = sys_txt_raw()
                                else:
                                    sys_val = sys_txt_raw
                            except Exception:
                                sys_val = sys_txt_raw
                            # Fallback to saved original system prompt when available
                            if not isinstance(sys_val, str):
                                try:
                                    sys_val = getattr(self, "_original_system_prompt", None)
                                except Exception:
                                    pass
                            sys_preview = summarize_and_redact_prompt(
                                sys_val if isinstance(sys_val, str) else str(sys_val),
                                max_length=preview_len_env,
                            )
                            attrs_sys = {
                                "model_id": self.model_id,
                                "attempt": getattr(attempt.retry_state, "attempt_number", 1),
                                "system_prompt_preview": sys_preview,
                            }
                            if debug_prompts:
                                attrs_sys["system_prompt_full"] = (
                                    sys_txt_raw
                                    if isinstance(sys_txt_raw, str)
                                    else str(sys_txt_raw)
                                )
                            tm.add_event("agent.system", attrs_sys)
                    except Exception:
                        pass

                    # Use adapter to convert pydantic-ai response to FlujoAgentResult
                    raw_agent_response = await asyncio.wait_for(
                        self._adapter.run(
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

                    # raw_agent_response is now a FlujoAgentResult
                    if isinstance(raw_agent_response, FlujoAgentResult):
                        # Check for error strings in output
                        if isinstance(
                            raw_agent_response.output, str
                        ) and raw_agent_response.output.startswith("Agent failed after"):
                            raise OrchestratorRetryError(raw_agent_response.output)

                        # Get usage info from FlujoAgentResult
                        agent_usage_info = raw_agent_response.usage()

                        # Get the actual output content to be processed
                        unpacked_output = raw_agent_response.output
                    else:
                        # Fallback for non-FlujoAgentResult responses (backward compatibility)
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

                    # Emit agent.input event (post-processor prompt snapshot)
                    try:
                        if tm is not None:
                            prompt_payload = processed_args[0] if processed_args else None
                            prompt_raw: Any = prompt_payload
                            try:
                                if isinstance(prompt_raw, PydanticBaseModel):
                                    prompt_raw = prompt_raw.model_dump(mode="json")
                                else:
                                    import dataclasses as _dc

                                    if _dc.is_dataclass(prompt_raw) and not isinstance(
                                        prompt_raw, type
                                    ):
                                        prompt_raw = _dc.asdict(prompt_raw)
                            except Exception:
                                pass
                            prompt_str = (
                                prompt_raw if isinstance(prompt_raw, str) else str(prompt_raw)
                            )
                            prompt_preview = summarize_and_redact_prompt(
                                prompt_str, max_length=preview_len_env
                            )
                            attrs_in = {
                                "model_id": self.model_id,
                                "attempt": getattr(attempt.retry_state, "attempt_number", 1),
                                "input_preview": prompt_preview,
                            }
                            if debug_prompts:
                                attrs_in["input_full"] = prompt_str
                            # Attach agent options snapshot for this attempt
                            try:
                                # Basic options exposed at wrapper level when set in ExecutorCore.
                                # We cannot directly access step.config here; ExecutorCore logs options separately.
                                attrs_in["options"] = (
                                    processed_kwargs.get("options")
                                    if isinstance(processed_kwargs, dict)
                                    else None
                                )
                            except Exception:
                                pass
                            tm.add_event("agent.input", attrs_in)
                    except Exception:
                        pass

                    if self.processors.output_processors:
                        processed = unpacked_output
                        for proc in self.processors.output_processors:
                            processed = await proc.process(processed, context_obj)
                        unpacked_output = processed

                    # If we already have a FlujoAgentResult, update its output with processed version
                    if isinstance(raw_agent_response, FlujoAgentResult):
                        # Update the output with processed version, preserve usage and cost
                        result = FlujoAgentResult(
                            output=unpacked_output,
                            usage=raw_agent_response.usage(),
                            cost_usd=raw_agent_response.cost_usd,
                            token_counts=raw_agent_response.token_counts,
                        )
                    else:
                        # Fallback: create FlujoAgentResult from legacy response
                        result = FlujoAgentResult(
                            output=unpacked_output,
                            usage=agent_usage_info,
                            cost_usd=None,
                            token_counts=None,
                        )

                    # Trace the response with preview and usage
                    try:
                        if tm is not None:
                            out_raw2: Any = result.output
                            try:
                                if isinstance(out_raw2, PydanticBaseModel):
                                    out_raw2 = out_raw2.model_dump(mode="json")
                                else:
                                    import dataclasses as _dc

                                    if _dc.is_dataclass(out_raw2) and not isinstance(
                                        out_raw2, type
                                    ):
                                        out_raw2 = _dc.asdict(out_raw2)
                            except Exception:
                                pass
                            out_str = out_raw2 if isinstance(out_raw2, str) else str(out_raw2)
                            out_prev = summarize_and_redact_prompt(
                                out_str, max_length=preview_len_env
                            )
                            attrs_out = {
                                "model_id": self.model_id,
                                "attempt": getattr(attempt.retry_state, "attempt_number", 1),
                                "response_preview": out_prev,
                            }
                            if debug_prompts:
                                attrs_out["response_full"] = out_str
                            tm.add_event("agent.response", attrs_out)
                            # Usage event
                            try:
                                u = result.usage()
                                if u is not None:
                                    # FlujoAgentUsage has input_tokens and output_tokens
                                    it = getattr(u, "input_tokens", None)
                                    ot = getattr(u, "output_tokens", None)
                                    cost = getattr(u, "cost_usd", None) or result.cost_usd
                                    tm.add_event(
                                        "agent.usage",
                                        {
                                            "input_tokens": it,
                                            "output_tokens": ot,
                                            "cost_usd": cost,
                                        },
                                    )
                            except Exception:
                                pass
                    except Exception:
                        pass

                    return result
        except RetryError as e:
            last_exc = e.last_attempt.exception()
            if last_exc is None:
                raise OrchestratorRetryError("Agent run failed with unknown error.") from e
            # Map pydantic-ai exceptions to Flujo exceptions (internal implementation detail)
            # Check for pydantic-ai specific exceptions and map them
            is_pydantic_ai_exception = False
            if isinstance(last_exc, ValidationError):
                is_pydantic_ai_exception = True
            elif isinstance(last_exc, ModelRetry):
                is_pydantic_ai_exception = True
            elif isinstance(last_exc, UnexpectedModelBehavior):
                is_pydantic_ai_exception = True

            # Phase 1 (AROS v2): catch provider JSON-mode failures (UnexpectedModelBehavior)
            # Map pydantic-ai exceptions internally but handle them as Flujo exceptions
            if is_pydantic_ai_exception and self.auto_repair:
                logfire.warn(
                    f"Agent validation failed. Initiating automated repair. Error: {last_exc}"
                )
                # Use module reference to allow monkeypatching in tests
                assert isinstance(last_exc, Exception)
                raw_output = agents_utils.get_raw_output_from_exception(last_exc)
                try:
                    cleaner = DeterministicRepairProcessor()
                    cleaned = await cleaner.process(raw_output)
                    validated = TypeAdapter(
                        _unwrap_type_adapter(self.target_output_type)
                    ).validate_json(cleaned)
                    logfire.info("Deterministic repair successful.")
                    # Return as FlujoAgentResult for consistent interface
                    return FlujoAgentResult(
                        output=validated, usage=None, cost_usd=None, token_counts=None
                    )
                except (ValidationError, ValueError, TypeError):
                    logfire.warn("Deterministic repair failed. Escalating to LLM repair.")
                try:
                    prompt_data = {
                        "json_schema": json.dumps(
                            TypeAdapter(
                                _unwrap_type_adapter(self.target_output_type)
                            ).json_schema(),
                            ensure_ascii=False,
                        ),
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
                            raise AgentIOValidationError(
                                f"Repair agent cannot fix output: {reasoning}"
                            )

                        # If not a repair error, validate against the target type
                        validated = TypeAdapter(
                            _unwrap_type_adapter(self.target_output_type)
                        ).validate_python(repair_response)
                        logfire.info("LLM repair successful.")
                        # Return as FlujoAgentResult for consistent interface
                        return FlujoAgentResult(
                            output=validated, usage=None, cost_usd=None, token_counts=None
                        )
                    except json.JSONDecodeError as decode_exc:
                        logfire.error(
                            f"LLM repair failed: Invalid JSON returned by repair agent: {decode_exc}\nRaw output: {repaired_str}"
                        )
                        raise AgentIOValidationError(
                            f"Agent validation failed: repair agent returned invalid JSON: {decode_exc}\nRaw output: {repaired_str}"
                        )
                    except (ValidationError, ValueError, TypeError) as repair_exc:
                        logfire.warn(f"LLM repair failed: {repair_exc}\nRaw output: {repaired_str}")
                        raise AgentIOValidationError(
                            f"Agent validation failed: schema validation error: {repair_exc}\nRaw output: {repaired_str}"
                        )
                except Exception as repair_agent_exc:
                    logfire.warn(f"Repair agent failed: {repair_agent_exc}")
                    raise AgentIOValidationError(
                        f"Repair agent execution failed: {repair_agent_exc}"
                    ) from repair_agent_exc
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
        except AgentIOValidationError:
            # Allow validation errors to propagate without re-wrapping so callers can classify them.
            raise
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
                raise ExecutionError(
                    f"Agent '{self._model_name}' execution failed: {error_type}({error_message})"
                )

    async def run_async(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent asynchronously with retry and timeout support.

        Returns:
            FlujoAgentResult: Vendor-agnostic result containing output and usage metrics.
            Note: Return type is Any for AsyncAgentProtocol compatibility, but this
            method always returns FlujoAgentResult at runtime.
        """
        return await self._run_with_retry(*args, **kwargs)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent (alias for run_async).

        Returns:
            FlujoAgentResult: Vendor-agnostic result containing output and usage metrics.
            Note: Return type is Any for AsyncAgentProtocol compatibility, but this
            method always returns FlujoAgentResult at runtime.
        """
        return await self.run_async(*args, **kwargs)

    # Structured output helpers (pydantic-ai centric; best-effort)
    def enable_structured_output(
        self, *, json_schema: dict[str, Any] | None = None, name: str | None = None
    ) -> None:
        """Enable structured output with JSON schema when supported.

        This sets a best-effort response_format hint; pydantic-ai/provider may
        ignore it when unsupported. Kept non-fatal by design.
        """
        try:
            if json_schema:
                nm = name or "step_output"
                self._structured_output_config = {
                    "type": "json_schema",
                    "json_schema": {"name": nm, "schema": json_schema},
                }
            else:
                self._structured_output_config = {"type": "json_object"}
        except Exception:
            self._structured_output_config = None


class TemplatedAsyncAgentWrapper(AsyncAgentWrapper[AgentInT, AgentOutT]):
    """
    Async wrapper that supports just-in-time system prompt rendering from a template
    using runtime context and previous step output.

    The wrapper temporarily overrides the underlying agent's system_prompt for a single
    run and restores it afterwards to keep agent instances stateless.
    """

    def __init__(
        self,
        agent: AgentLike,
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
        self._prompt_lock: Optional[asyncio.Lock] = None

    def _get_prompt_lock(self) -> asyncio.Lock:
        """Lazily create the prompt lock on first access."""
        if self._prompt_lock is None:
            self._prompt_lock = asyncio.Lock()
        return self._prompt_lock

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

                        # Prepare template kwargs
                        template_kwargs = {
                            "context": ctx_proxy,
                            "previous_step": previous_step,
                            "steps": steps_wrapped,
                        }

                        # Add resume_input if HITL history exists
                        try:
                            if (
                                context
                                and hasattr(context, "hitl_history")
                                and context.hitl_history
                            ):
                                template_kwargs["resume_input"] = context.hitl_history[
                                    -1
                                ].human_response
                        except Exception:
                            pass

                        resolved_vars[key] = format_prompt(value_template, **template_kwargs)
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

                # Prepare final template kwargs
                final_kwargs = {
                    **resolved_vars,
                    "context": ctx_proxy,
                    "previous_step": previous_step,
                    "steps": steps_wrapped,
                }

                # Add resume_input if HITL history exists
                try:
                    if context and hasattr(context, "hitl_history") and context.hitl_history:
                        final_kwargs["resume_input"] = context.hitl_history[-1].human_response
                except Exception:
                    pass

                final_system_prompt = format_prompt(
                    self.system_prompt_template,
                    **final_kwargs,
                )
            except Exception:
                final_system_prompt = self.system_prompt_template

            # Trace the resolved variables for debugging (redacted)
            try:
                tm = get_active_trace_manager()
                if tm is not None:
                    import os as _os
                    from ..utils.redact import summarize_and_redact_prompt as _sum

                    full_flag = _os.getenv("FLUJO_DEBUG_PROMPTS") == "1"
                    try:
                        prev_len = int(_os.getenv("FLUJO_TRACE_PREVIEW_LEN", "1000"))
                    except Exception:
                        prev_len = 1000
                    vars_preview = {
                        k: (_sum(str(v), max_length=prev_len) if not full_flag else str(v))
                        for k, v in (resolved_vars or {}).items()
                    }
                    tm.add_event("agent.system.vars", {"vars": vars_preview})
            except Exception:
                pass

            # Temporarily override system prompt with concurrency protection
            async with self._get_prompt_lock():
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

    The wrapper uses an internal adapter pattern to convert pydantic-ai responses
    to FlujoAgentResult, providing vendor-agnostic results to Flujo's orchestration layer.

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

    Returns
    -------
    AsyncAgentWrapper
        Wrapper that returns FlujoAgentResult (vendor-agnostic interface).
        The adapter pattern is used internally to isolate pydantic-ai specifics.
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

    wrapper: AsyncAgentWrapper[Any, Any] = AsyncAgentWrapper(
        agent,
        max_retries=max_retries,
        timeout=timeout,
        model_name=model,
        processors=final_processors,
        auto_repair=auto_repair,
    )
    try:
        # Preserve original system prompt text for telemetry when provider hides it behind a callable
        setattr(wrapper, "_original_system_prompt", system_prompt)
    except Exception:
        pass
    return wrapper


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

    The wrapper uses an internal adapter pattern to convert pydantic-ai responses
    to FlujoAgentResult, providing vendor-agnostic results to Flujo's orchestration layer.

    Parameters
    ----------
    model : str
        The model identifier (e.g., "openai:gpt-4o")
    template_string : str
        Template string for system prompt with variable placeholders
    variables_spec : Optional[dict[str, Any]]
        Variable specifications for template rendering
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

    Returns
    -------
    TemplatedAsyncAgentWrapper
        Wrapper that returns FlujoAgentResult (vendor-agnostic interface).
        The adapter pattern is used internally to isolate pydantic-ai specifics.
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
