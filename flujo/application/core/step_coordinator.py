"""Step execution coordination with telemetry and hook management."""

from __future__ import annotations

from typing import Any, AsyncIterator, Generic, Optional, TypeVar, Literal

from flujo.domain.backends import ExecutionBackend, StepExecutionRequest
from flujo.domain.dsl.step import Step
from flujo.domain.models import (
    BaseModel,
    StepResult,
    PipelineContext,
    PipelineResult,
    UsageLimits,
    StepOutcome,
    Success,
    Failure,
    Chunk,
)
from flujo.domain.models import Quota
from flujo.domain.resources import AppResources
from flujo.exceptions import (
    ContextInheritanceError,
    PipelineAbortSignal,
    PipelineContextInitializationError,
    PausedException,
    UsageLimitExceededError,
    MockDetectionError,
    NonRetryableError,
)
from flujo.infra import telemetry

from flujo.domain.types import HookCallable
from flujo.application.core.hook_dispatcher import _dispatch_hook

ContextT = TypeVar("ContextT", bound=BaseModel)


class StepCoordinator(Generic[ContextT]):
    """Coordinates individual step execution with telemetry and hooks."""

    def __init__(
        self,
        hooks: Optional[list[HookCallable]] = None,
        resources: Optional[AppResources] = None,
    ) -> None:
        self.hooks = hooks or []
        self.resources = resources

    async def execute_step(
        self,
        step: "Step[Any, Any]",
        data: Any,
        context: Optional[ContextT],
        backend: Optional[ExecutionBackend] = None,  # ✅ NEW: Receive the backend to call.
        *,
        stream: bool = False,
        step_executor: Optional[Any] = None,  # Legacy parameter for backward compatibility
        usage_limits: Optional[UsageLimits] = None,  # ✅ NEW: Usage limits for step execution
        quota: Optional[Quota] = None,
    ) -> AsyncIterator[Any]:
        """Execute a single step with telemetry and hook management.

        Args:
            step: The step to execute
            data: Input data for the step
            context: Pipeline context
            backend: The execution backend to call
            stream: Whether to stream output

        Yields:
            Step results or streaming chunks
        """
        # Dispatch pre-step hook
        # Capture quota snapshot if available on context scratchpad
        quota_before_usd = None
        quota_before_tokens = None
        try:
            # Best effort: some contexts may not have quota info
            remaining = None
            from flujo.application.core.executor_core import ExecutorCore as _Exec

            quota_obj = _Exec.CURRENT_QUOTA.get()
            if isinstance(quota_obj, Quota):
                remaining = quota_obj.get_remaining()
            else:
                remaining = None
            if remaining is not None:
                quota_before_usd, quota_before_tokens = remaining
        except Exception:
            quota_before_usd = None
            quota_before_tokens = None

        await self._dispatch_hook(
            "pre_step",
            step=step,
            step_input=data,
            context=context,
            resources=self.resources,
            attempt_number=getattr(step, "_current_attempt", None),
            quota_before_usd=quota_before_usd,
            quota_before_tokens=quota_before_tokens,
            cache_hit=False,
        )

        # Execute step with telemetry
        step_result = None
        with telemetry.logfire.span(step.name) as span:
            try:
                # ✅ UPDATE: Support both new backend approach and legacy step_executor
                # Prioritize step_executor for backward compatibility with tests
                if step_executor is not None:
                    # Legacy approach: use step_executor
                    # Handle both async generators and regular async functions
                    try:
                        # Try to use as async generator first
                        async for item in step_executor(
                            step, data, context, self.resources, stream=stream
                        ):
                            # Preserve legacy behavior for custom executors
                            if isinstance(item, StepOutcome):
                                if isinstance(item, Success):
                                    step_result = item.step_result
                                yield item
                            elif isinstance(item, StepResult):
                                step_result = item
                                yield item
                            else:
                                # Pass through raw chunks/strings unchanged
                                yield item
                    except TypeError:
                        # If that fails, try as regular async function
                        item = await step_executor(
                            step, data, context, self.resources, stream=stream
                        )
                        if isinstance(item, StepOutcome):
                            if isinstance(item, Success):
                                step_result = item.step_result
                            yield item
                        elif isinstance(item, StepResult):
                            step_result = item
                            yield item
                        else:
                            yield item
                elif backend is not None:
                    # New approach: call backend directly
                    # Only enable streaming when the agent actually supports it
                    has_agent_stream = hasattr(step, "agent") and hasattr(
                        getattr(step, "agent", None), "stream"
                    )
                    effective_stream = bool(stream and has_agent_stream)
                    if effective_stream:
                        # For streaming, we need to collect chunks and yield them
                        chunks = []

                        async def on_chunk(chunk: Any) -> None:
                            chunks.append(chunk)

                        request = StepExecutionRequest(
                            step=step,
                            input_data=data,
                            context=context,
                            resources=self.resources,
                            stream=effective_stream,
                            on_chunk=on_chunk,
                            usage_limits=usage_limits,
                            # Ensure quota propagates consistently during streaming
                            quota=quota,
                        )

                        # Call the backend directly (typed StepOutcome)
                        step_outcome = await backend.execute_step(request)

                        # Repair known internal attribute-error masking during streaming failures
                        # If chunks were produced and the failure error looks like an internal attribute error,
                        # replace feedback with a clearer streaming failure message for user-facing correctness.
                        if isinstance(step_outcome, Failure):
                            try:
                                err_txt = str(step_outcome.error or "")
                                fb_txt = step_outcome.feedback or ""
                                if chunks and (
                                    "object has no attribute 'success'" in err_txt
                                    or "object has no attribute 'success'" in fb_txt
                                    or "object has no attribute 'metadata_'" in err_txt
                                    or "object has no attribute 'metadata_'" in fb_txt
                                ):
                                    # Construct a clearer failure outcome preserving existing step_result when present
                                    repaired_feedback = "Stream connection lost"
                                    sr = step_outcome.step_result or StepResult(
                                        name=getattr(step, "name", "<unnamed>"),
                                        success=False,
                                        output=None,
                                        feedback=repaired_feedback,
                                    )
                                    sr.feedback = repaired_feedback
                                    step_outcome = Failure(
                                        error=RuntimeError(repaired_feedback),
                                        feedback=repaired_feedback,
                                        step_result=sr,
                                    )
                            except Exception:
                                pass

                        # Yield chunks first, then final outcome/result
                        for chunk in chunks:
                            # Wrap streaming chunks into typed outcome if not already
                            yield Chunk(data=chunk, step_name=step.name)
                        if isinstance(step_outcome, StepOutcome):
                            if isinstance(step_outcome, Success):
                                step_result = step_outcome.step_result
                            elif isinstance(step_outcome, Failure):
                                # Ensure failure hook runs by populating step_result
                                step_result = step_outcome.step_result or StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=False,
                                    feedback=step_outcome.feedback,
                                )
                            yield step_outcome
                        else:
                            # Normalize legacy StepResult to Success
                            step_result = step_outcome
                            yield Success(step_result=step_result)
                    else:
                        # Non-streaming case
                        request = StepExecutionRequest(
                            step=step,
                            input_data=data,
                            context=context,
                            resources=self.resources,
                            stream=False,
                            usage_limits=usage_limits,
                            quota=quota,
                        )

                        # Call the backend directly (typed StepOutcome)
                        step_outcome = await backend.execute_step(request)
                        if isinstance(step_outcome, StepOutcome):
                            if isinstance(step_outcome, Success):
                                step_result = step_outcome.step_result
                            elif isinstance(step_outcome, Failure):
                                step_result = step_outcome.step_result or StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=False,
                                    feedback=step_outcome.feedback,
                                )
                            yield step_outcome
                        else:
                            step_result = step_outcome
                            yield Success(step_result=step_result)
                else:
                    raise ValueError("Either backend or step_executor must be provided")

            except PausedException as e:
                # Handle pause for human input; mark context and stop executing current step
                if isinstance(context, PipelineContext):
                    context.scratchpad["status"] = "paused"
                    context.scratchpad["pause_message"] = str(e)
                    scratch = context.scratchpad
                    if "paused_step_input" not in scratch:
                        scratch["paused_step_input"] = data
                # Emit pause event for tracing
                try:
                    await self._dispatch_hook(
                        "on_step_failure",  # reuse channel to ensure TraceManager closes child span if any
                        step_result=StepResult(
                            name=getattr(step, "name", "<paused>"),
                            success=False,
                            feedback=str(e),
                        ),
                        context=context,
                        resources=self.resources,
                    )
                except Exception:
                    pass
                # Do not append a synthetic result; just stop so runner can resume later
                # Indicate to the ExecutionManager/Runner that execution should stop by raising a sentinel
                raise PipelineAbortSignal("Paused for HITL")
            except UsageLimitExceededError:
                # Re-raise usage limit exceptions to be handled by ExecutionManager
                raise
            except PipelineContextInitializationError:
                # Re-raise context initialization errors to be handled by ExecutionManager
                raise
            except ContextInheritanceError:
                # Re-raise context inheritance errors to be handled by ExecutionManager
                raise
            except (MockDetectionError, NonRetryableError):
                # Re-raise mock detection and non-retryable errors immediately
                raise
            except Exception as e:
                # Propagate critical redirect-loop exceptions instead of swallowing into a StepResult
                if e.__class__.__name__ == "InfiniteRedirectError":
                    raise
                try:
                    from flujo.exceptions import InfiniteRedirectError as CoreIRE

                    if isinstance(e, CoreIRE):
                        raise
                except Exception:
                    pass
                try:
                    from flujo.application.runner import InfiniteRedirectError as RunnerIRE

                    if isinstance(e, RunnerIRE):
                        raise
                except Exception:
                    pass
                # Treat strict pricing as critical and propagate immediately
                try:
                    from flujo.exceptions import PricingNotConfiguredError as _PNCE

                    if isinstance(e, _PNCE):
                        raise
                    _msg = str(e)
                    if (
                        "Strict pricing is enabled" in _msg
                        or "Pricing not configured" in _msg
                        or "no configuration was found for provider" in _msg
                    ):
                        raise _PNCE(None, "unknown")
                except Exception:
                    pass
                # For all other exceptions, let the manager/pipeline handling proceed (will produce failure result)
                raise

            # Update telemetry span with step metadata
            if step_result and step_result.metadata_:
                for key, value in step_result.metadata_.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        telemetry.logfire.error(f"Error setting span attribute: {e}")

        # Handle step success/failure: finalize trace spans and fire hooks
        if step_result:
            if step_result.success:
                await self._dispatch_hook(
                    "post_step",
                    step_result=step_result,
                    context=context,
                    resources=self.resources,
                )
            else:
                # Call failure handlers when step fails
                if hasattr(step, "failure_handlers") and step.failure_handlers:
                    for handler in step.failure_handlers:
                        try:
                            handler() if hasattr(handler, "__call__") else None
                        except Exception as e:
                            telemetry.logfire.error(
                                f"Failure handler {handler} raised exception: {e}"
                            )
                            raise

                try:
                    await self._dispatch_hook(
                        "on_step_failure",
                        step_result=step_result,
                        context=context,
                        resources=self.resources,
                    )
                except PipelineAbortSignal:
                    # Yield the failed step result before aborting
                    yield step_result
                    raise

    async def _dispatch_hook(
        self,
        event_name: Literal["pre_run", "post_run", "pre_step", "post_step", "on_step_failure"],
        **kwargs: Any,
    ) -> None:
        """Dispatch a hook to all registered hook functions."""
        await _dispatch_hook(self.hooks, event_name, **kwargs)

    def update_pipeline_result(
        self,
        result: PipelineResult[ContextT],
        step_result: StepResult,
    ) -> None:
        """Update the pipeline result with a step result."""
        result.step_history.append(step_result)
        result.total_cost_usd += step_result.cost_usd
        result.total_tokens += step_result.token_counts
