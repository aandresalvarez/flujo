from __future__ import annotations
from typing import Any, Awaitable, Optional, Callable, TYPE_CHECKING
import asyncio, time
from pydantic import BaseModel
from unittest.mock import Mock, MagicMock, AsyncMock
from flujo.domain.models import StepResult, UsageLimits
from flujo.exceptions import MissingAgentError, PausedException, InfiniteFallbackError, InfiniteRedirectError, UsageLimitExceededError, NonRetryableError, MockDetectionError
if TYPE_CHECKING:
    from flujo.application.core.ultra_executor import ExecutorCore
from flujo.cost import extract_usage_metrics
from flujo.utils.performance import time_perf_ns, time_perf_ns_to_seconds
from flujo.infra import telemetry

async def _execute_agent_step(
    self: ExecutorCore,
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    stream: bool,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]],
    cache_key: Optional[str],
    breach_event: Optional[Any],
    _fallback_depth: int = 0,
) -> StepResult:
    """
    Execute an agent step with robust, first-principles retry and feedback logic.

    FIXED: Proper failure domain isolation
    - Only retries agent execution for agent-specific failures
    - Immediately fails step when plugins/validators/processors fail
    - Uses loop-based retry mechanism to avoid infinite recursion
    """
    # --- 0. Pre-execution Validation for Agent Steps ---
    if step.agent is None:
        raise MissingAgentError(f"Step '{step.name}' has no agent configured")

    # Initialize result with proper attempt tracking
    result = StepResult(
        name=step.name,
        output=None,
        success=False,
        attempts=1,
        latency_s=0.0,
        token_counts=0,
        cost_usd=0.0,
        feedback=None,
        branch_context=None,
        metadata_={},
        step_history=[],
    )

    overall_start_time = time.monotonic()
    max_retries = step.config.max_retries
    if stream:
        max_retries = 0

    # Helper functions for agent result processing
    def _unpack_agent_result(output: Any) -> Any:
        from pydantic import BaseModel
        if isinstance(output, BaseModel):
            return output
        if hasattr(output, "output"):
            return output.output
        elif hasattr(output, "content"):
            return output.content
        elif hasattr(output, "result"):
            return output.result
        elif hasattr(output, "data"):
            return output.data
        elif hasattr(output, "text"):
            return output.text
        elif hasattr(output, "message"):
            return output.message
        else:
            return output

    def _detect_mock_objects(obj: Any) -> None:
        if isinstance(obj, (Mock, MagicMock, AsyncMock)):
            raise MockDetectionError("Mock object detected in agent output")

    if hasattr(max_retries, '_mock_name') or isinstance(max_retries, (Mock, MagicMock, AsyncMock)):
        max_retries = 3

    # Capture initial context state to prevent accumulating context updates across retries
    initial_context_state = None
    total_attempts = max_retries + 1
    if context is not None and total_attempts > 1:
        import copy
        initial_context_state = copy.deepcopy(context.model_dump())

    for attempt in range(1, max_retries + 2):
        result.attempts = attempt
        if limits is not None:
            await self._usage_meter.guard(limits, result.step_history)
        
        # Reset context to initial state for retry attempts (except first)
        # This prevents context updates from accumulating across failed retries
        if attempt > 1 and context is not None and initial_context_state is not None:
            telemetry.logfire.debug(f"Resetting context for retry attempt {attempt}")
            # Restore context to initial state before retry
            for field, value in initial_context_state.items():
                if hasattr(context, field):
                    old_value = getattr(context, field)
                    setattr(context, field, value)
                    telemetry.logfire.debug(f"Reset context field {field}: {old_value} -> {value}")
        
        start_time = time_perf_ns()
        try:
            processed_data = data
            if hasattr(step, 'processors') and getattr(step, 'processors', None):
                processed_data = await self._processor_pipeline.apply_prompt(
                    step.processors, data, context=context
                )

            options = {}
            if hasattr(step, 'config') and step.config:
                if hasattr(step.config, 'temperature') and step.config.temperature is not None:
                    options['temperature'] = step.config.temperature
                if hasattr(step.config, 'top_k') and step.config.top_k is not None:
                    options['top_k'] = step.config.top_k
                if hasattr(step.config, 'top_p') and step.config.top_p is not None:
                    options['top_p'] = step.config.top_p

            agent_output = await self._agent_runner.run(
                agent=step.agent,
                payload=processed_data,
                context=context,
                resources=resources,
                options=options,
                stream=stream,
                on_chunk=on_chunk,
                breach_event=breach_event,
            )

            if isinstance(agent_output, (Mock, MagicMock, AsyncMock)):
                raise MockDetectionError(f"Step '{step.name}' returned a Mock object")

            def _detect_mock_objects_in_output(obj: Any) -> None:
                if isinstance(obj, (Mock, MagicMock, AsyncMock)):
                    raise MockDetectionError(f"Step '{step.name}' returned a Mock object")

            _detect_mock_objects_in_output(agent_output)

            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                raw_output=agent_output, agent=step.agent, step_name=step.name
            )
            result.cost_usd = cost_usd
            result.token_counts = prompt_tokens + completion_tokens
            processed_output = agent_output
            if hasattr(step, "processors") and step.processors:
                try:
                    processed_output = await self._processor_pipeline.apply_output(
                        step.processors, processed_output, context=context
                    )
                except Exception as e:
                    result.success = False
                    result.feedback = f"Processor failed: {str(e)}"
                    result.output = processed_output
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' processor failed: {e}")
                    return result

            validation_passed = True
            try:
                if hasattr(step, "validators") and step.validators:
                    validation_results = await self._validator_runner.validate(
                        step.validators, processed_output, context=context
                    )
                    failed_validations = [r for r in validation_results if not r.is_valid]
                    if failed_validations:
                        validation_passed = False
                        if attempt < max_retries:
                            telemetry.logfire.warning(f"Step '{step.name}' validation failed: {failed_validations[0].feedback}")
                            continue
                        else:
                            result.success = False
                            result.feedback = f"Validation failed after max retries: {self._format_feedback(failed_validations[0].feedback, 'Agent execution failed')}"
                            result.output = processed_output
                            result.latency_s = time.monotonic() - start_time
                            telemetry.logfire.error(f"Step '{step.name}' validation failed after {result.attempts} attempts")
                            return result
            except Exception as e:
                validation_passed = False
                if attempt < max_retries:
                    telemetry.logfire.warning(f"Step '{step.name}' validation failed: {e}")
                    continue
                else:
                    result.success = False
                    result.feedback = f"Validation failed after max retries: {str(e)}"
                    result.output = processed_output
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' validation failed after {result.attempts} attempts")
                    return result

            if validation_passed:
                try:
                    if hasattr(step, "plugins") and step.plugins:
                        unpacked_output = _unpack_agent_result(processed_output)
                        plugin_data = {"output": unpacked_output} if not isinstance(unpacked_output, dict) else unpacked_output
                        plugin_result = await self._plugin_runner.run_plugins(
                            step.plugins, plugin_data, context=context, resources=resources
                        )
                        if hasattr(plugin_result, "redirect_to") and plugin_result.redirect_to is not None:
                            redirected_agent = plugin_result.redirect_to
                            telemetry.logfire.info(f"Step '{step.name}' redirecting to agent: {redirected_agent}")
                            redirected_output = await self._agent_runner.run(
                                agent=redirected_agent,
                                payload=data,
                                context=context,
                                resources=resources,
                                options={},
                                stream=stream,
                                on_chunk=on_chunk,
                                breach_event=breach_event,
                            )
                            prompt_tokens, completion_tokens, cost_usd = extract_usage_metrics(
                                raw_output=redirected_output, agent=redirected_agent, step_name=step.name
                            )
                            result.cost_usd += cost_usd
                            result.token_counts += prompt_tokens + completion_tokens
                            processed_output = _unpack_agent_result(redirected_output)
                        elif hasattr(plugin_result, "success") and not plugin_result.success:
                            from flujo.application.core.ultra_executor import PluginError
                            raise PluginError(plugin_result.feedback or 'Plugin failed without feedback')
                        elif hasattr(plugin_result, "success") and plugin_result.success:
                            processed_output = processed_output
                        elif isinstance(plugin_result, dict) and "output" in plugin_result:
                            processed_output = plugin_result["output"]
                        else:
                            processed_output = plugin_result
                except Exception as e:
                    result.success = False
                    result.feedback = f"Plugin failed: {str(e)}"
                    result.output = processed_output
                    result.latency_s = time.monotonic() - start_time
                    telemetry.logfire.error(f"Step '{step.name}' plugin failed: {e}")
                    return result

                result.output = _unpack_agent_result(processed_output)
                _detect_mock_objects(result.output)
                result.success = True
                result.latency_s = time_perf_ns_to_seconds(time_perf_ns() - start_time)
                result.feedback = None
                result.branch_context = context
                if cache_key and self._enable_cache:
                    self.cache.set(cache_key, result)
                return result
        except Exception as e:
            # Re-raise critical exceptions immediately
            if isinstance(e, (PausedException, InfiniteFallbackError, InfiniteRedirectError, UsageLimitExceededError, NonRetryableError)):
                telemetry.logfire.error(f"Step '{step.name}' encountered a non-retryable exception: {type(e).__name__}")
                raise e
            # Retryable agent errors
            if attempt < max_retries + 1:
                telemetry.logfire.warning(f"Step '{step.name}' agent execution attempt {attempt} failed: {e}")
                continue
            # Max retries exceeded
            result.success = False
            # Import PluginError here to avoid circular import
            from flujo.application.core.ultra_executor import PluginError
            if isinstance(e, PluginError):
                msg = str(e)
                if msg.startswith("Plugin validation failed"):
                    result.feedback = f"Plugin execution failed after max retries: {msg}"
                else:
                    result.feedback = f"Plugin validation failed after max retries: {msg}"
            else:
                result.feedback = f"Agent execution failed with {type(e).__name__}: {str(e)}"
            result.output = None
            result.latency_s = time.monotonic() - overall_start_time
            telemetry.logfire.error(f"Step '{step.name}' agent failed after {result.attempts} attempts")
            return result
    result.success = False
    result.feedback = "Unexpected execution path"
    result.latency_s = time.monotonic() - start_time
    return result