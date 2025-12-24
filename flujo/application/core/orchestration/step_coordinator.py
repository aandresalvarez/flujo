"""Step execution coordination with telemetry and hook management."""

from __future__ import annotations

from typing import AsyncIterator, Generic, Literal, Optional, TypeVar

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
    BackgroundLaunched,
)
from flujo.domain.models import Paused as _Paused
from flujo.domain.models import Quota
from flujo.domain.resources import AppResources
from flujo.exceptions import (
    ContextInheritanceError,
    PipelineAbortSignal,
    PipelineContextInitializationError,
    PausedException,
    UsageLimitExceededError,
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
        step: "Step[object, object]",
        data: object,
        context: Optional[ContextT],
        backend: Optional[ExecutionBackend] = None,  # ✅ NEW: Receive the backend to call.
        *,
        stream: bool = False,
        usage_limits: Optional[UsageLimits] = None,  # ✅ NEW: Usage limits for step execution
        quota: Optional[Quota] = None,
    ) -> AsyncIterator[object]:
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
        # Capture quota snapshot if available on provided quota
        quota_before_usd = None
        quota_before_tokens = None
        try:
            if quota is not None:
                remaining = quota.get_remaining()
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
                if backend is not None:
                    # New approach: call backend directly
                    # Only enable streaming when the agent actually supports it
                    has_agent_stream = hasattr(step, "agent") and hasattr(
                        getattr(step, "agent", None), "stream"
                    )
                    effective_stream = bool(stream and has_agent_stream)
                    if effective_stream:
                        # For streaming, we need to collect chunks and yield them
                        chunks: list[object] = []
                        _last_chunk: object = object()

                        async def on_chunk(chunk: object) -> None:
                            nonlocal _last_chunk
                            # Deduplicate consecutive identical chunks to avoid double-emission
                            if chunk == _last_chunk:
                                return
                            chunks.append(chunk)
                            _last_chunk = chunk

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
                            try:
                                import os as _os

                                if _os.getenv("FLUJO_TRACE_EXTRA", "") == "1":
                                    from flujo.tracing.manager import (
                                        get_active_trace_manager as _get_tm,
                                    )

                                    tm = _get_tm()
                                    if tm is not None:
                                        tm.add_event(
                                            "coordinator.outcome",
                                            {
                                                "kind": type(step_outcome).__name__,
                                                "step": getattr(step, "name", "<unnamed>"),
                                            },
                                        )
                            except Exception:
                                pass
                            if isinstance(step_outcome, Success):
                                step_result = step_outcome.step_result
                                # Normalize loop semantics: reaching max_loops is considered
                                # a successful completion for refine-style loops so that the
                                # post-loop mapper can run and inspect the critic's final check.
                                try:
                                    meta = getattr(step_result, "metadata_", {}) or {}
                                    if meta.get("exit_reason") == "max_loops":
                                        step_result.success = True
                                except Exception:
                                    pass
                            elif isinstance(step_outcome, Failure):
                                # Ensure failure hook runs by populating step_result
                                step_result = step_outcome.step_result or StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=False,
                                    feedback=step_outcome.feedback,
                                )
                            yield step_outcome
                        else:
                            # Normalize legacy StepResult to typed outcome; if it indicates
                            # failure, convert to Failure so the manager short-circuits.
                            from flujo.domain.models import StepResult as _SR

                            if isinstance(step_outcome, _SR):
                                step_result = step_outcome
                                try:
                                    if getattr(step_result, "success", True) is False:
                                        yield Failure(
                                            error=Exception(step_result.feedback or "step failed"),
                                            feedback=step_result.feedback,
                                            step_result=step_result,
                                        )
                                        return
                                except Exception:
                                    pass
                                yield Success(step_result=step_result)
                            else:
                                # Unknown legacy value -> synthesize a failure
                                synthesized = StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=False,
                                    feedback="Agent produced no terminal outcome",
                                )
                                yield Failure(
                                    error=RuntimeError("Agent produced no terminal outcome"),
                                    feedback=synthesized.feedback,
                                    step_result=synthesized,
                                )
                    else:
                        # Non-streaming case
                        request = StepExecutionRequest(
                            step=step,
                            input_data=data,
                            context=context,
                            resources=self.resources,
                            stream=stream,
                            usage_limits=usage_limits,
                            quota=quota,
                        )

                        # Call the backend directly (typed StepOutcome)
                        step_outcome = await backend.execute_step(request)
                        if isinstance(step_outcome, StepOutcome):
                            try:
                                import os as _os

                                if _os.getenv("FLUJO_TRACE_EXTRA", "") == "1":
                                    from flujo.tracing.manager import (
                                        get_active_trace_manager as _get_tm,
                                    )

                                    tm = _get_tm()
                                    if tm is not None:
                                        tm.add_event(
                                            "coordinator.outcome",
                                            {
                                                "kind": type(step_outcome).__name__,
                                                "step": getattr(step, "name", "<unnamed>"),
                                            },
                                        )
                            except Exception:
                                pass
                            if isinstance(step_outcome, Success):
                                step_result = step_outcome.step_result
                                # Normalize loop/reporting semantics: if a policy produced
                                # a Success outcome with a StepResult marked success=False
                                # (e.g., loop exited due to max iterations), convert it to
                                # a Failure outcome so the manager short-circuits and does
                                # not execute downstream steps (like mappers).
                                try:
                                    if (
                                        step_result is not None
                                        and getattr(step_result, "success", True) is False
                                    ):
                                        yield Failure(
                                            error=Exception(step_result.feedback or "step failed"),
                                            feedback=step_result.feedback,
                                            step_result=step_result,
                                        )
                                        return
                                except Exception:
                                    pass
                                yield step_outcome
                            elif isinstance(step_outcome, Failure):
                                step_result = step_outcome.step_result or StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=False,
                                    feedback=step_outcome.feedback,
                                )
                                yield step_outcome
                            elif isinstance(step_outcome, _Paused):
                                # Normalize Paused control flow to PipelineAbortSignal and mark context
                                try:
                                    if isinstance(context, PipelineContext):
                                        context.status = "paused"
                                        # Use plain message for backward compatibility
                                        msg = getattr(step_outcome, "message", "Paused for HITL")
                                        context.pause_message = (
                                            msg if isinstance(msg, str) else str(msg)
                                        )
                                        if context.paused_step_input is None:
                                            context.paused_step_input = data
                                except Exception:
                                    pass
                                raise PipelineAbortSignal("Paused for HITL")
                            elif isinstance(step_outcome, BackgroundLaunched):
                                step_result = StepResult(
                                    name=getattr(step, "name", "<unnamed>"),
                                    success=True,
                                    output=data,  # Pass through input data to next step
                                    feedback=f"Launched in background (task_id={step_outcome.task_id})",
                                    metadata_={"background_task_id": step_outcome.task_id},
                                )
                                yield step_outcome
                            else:
                                # Unknown control outcome: propagate as abort

                                raise PipelineAbortSignal("Paused for HITL")
                        else:
                            # Backend returned a legacy value or invalid.
                            # If it's a proper StepResult, normalize to Success without re-executing.
                            from flujo.domain.models import StepResult as _SR

                            if isinstance(step_outcome, _SR):
                                step_result = step_outcome
                                yield Success(step_result=step_result)
                            else:
                                # Fall back to executing the step directly via ExecutorCore to recover.
                                try:
                                    from ..executor_core import ExecutorCore as _Core
                                    from ..types import ExecutionFrame as _Frame

                                    core2 = _Core()
                                    frame2 = _Frame(
                                        step=step,
                                        data=data,
                                        context=context,
                                        resources=self.resources,
                                        limits=usage_limits,
                                        quota=quota,
                                        stream=False,
                                        on_chunk=None,
                                        context_setter=lambda _r, _c: None,
                                    )
                                    out2 = await core2.execute(frame2)
                                    if isinstance(out2, StepOutcome):
                                        yield out2
                                    else:
                                        # Legacy: normalize StepResult to Success
                                        step_result = out2
                                        yield Success(step_result=step_result)
                                except Exception as e:
                                    # Control-flow exceptions must propagate; never coerce into failures.
                                    if isinstance(e, (PausedException, PipelineAbortSignal)):
                                        raise
                                    try:
                                        telemetry.logfire.error(
                                            f"Coordinator internal execute failed: {e}"
                                        )
                                    except Exception:
                                        pass
                                    # As a last resort, synthesize a failure outcome
                                    synthesized = StepResult(
                                        name=getattr(step, "name", "<unnamed>"),
                                        success=False,
                                        feedback="Agent produced no terminal outcome",
                                    )
                                    yield Failure(
                                        error=RuntimeError("Agent produced no terminal outcome"),
                                        feedback=synthesized.feedback,
                                        step_result=synthesized,
                                    )
                else:
                    raise ValueError("backend must be provided")

            except PausedException as e:
                # Handle pause for human input; mark context and stop executing current step
                if isinstance(context, PipelineContext):
                    context.status = "paused"
                    # Use plain message for backward compatibility (tests expect plain message)
                    # Only set if not already set (loop policy or recipe may have set it already)
                    if context.pause_message is None:
                        context.pause_message = getattr(e, "message", "")
                    # If already set, preserve it (loop policy/recipe already set it correctly)
                    if context.paused_step_input is None:
                        context.paused_step_input = data
                # Indicate to the ExecutionManager/Runner that execution should stop by raising a sentinel
                raise PipelineAbortSignal("Paused for HITL") from e
            except UsageLimitExceededError:
                # Re-raise usage limit exceptions to be handled by ExecutionManager
                raise
            except PipelineContextInitializationError:
                # Re-raise context initialization errors to be handled by ExecutionManager
                raise
            except ContextInheritanceError:
                # Re-raise context inheritance errors to be handled by ExecutionManager
                raise
            except NonRetryableError:
                # Re-raise non-retryable errors immediately
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
                            if callable(handler):
                                handler()
                            try:
                                if isinstance(step_result, StepResult):
                                    meta = getattr(step_result, "metadata_", None)
                                    if not isinstance(meta, dict):
                                        step_result.metadata_ = {}
                                        meta = step_result.metadata_
                                    meta["failure_handlers_ran"] = True
                            except Exception:
                                pass
                        except Exception as e:
                            telemetry.logfire.error(
                                f"Failure handler {handler} raised exception: {e}"
                            )
                            raise
                dispatch_needed = True
                try:
                    if isinstance(step_result, StepResult):
                        meta = getattr(step_result, "metadata_", None)
                        if not isinstance(meta, dict):
                            step_result.metadata_ = {}
                            meta = step_result.metadata_
                        if meta.get("on_step_failure_dispatched"):
                            dispatch_needed = False
                        else:
                            meta["on_step_failure_dispatched"] = True
                except Exception:
                    dispatch_needed = True

                if dispatch_needed:
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
        else:
            # Safety net: ensure every step yields a terminal outcome
            try:
                synthesized = StepResult(
                    name=getattr(step, "name", "<unnamed>"),
                    success=False,
                    output=None,
                    feedback="Agent produced no terminal outcome",
                )
                try:
                    import os as _os

                    if _os.getenv("FLUJO_TRACE_EXTRA", "") == "1":
                        from flujo.tracing.manager import get_active_trace_manager as _get_tm

                        tm = _get_tm()
                        if tm is not None:
                            tm.add_event(
                                "coordinator.no_terminal",
                                {"step": getattr(step, "name", "<unnamed>")},
                            )
                except Exception:
                    pass
                # Do not fire step-level hooks here; ExecutionManager will treat this as a Failure
                # and route through its failure path which already dispatches on_step_failure.
                yield Failure(
                    error=RuntimeError("Agent produced no terminal outcome"),
                    feedback=synthesized.feedback,
                    step_result=synthesized,
                )
            except Exception:
                # As a last resort, do nothing — ExecutionManager has its own safety net.
                pass

    async def _dispatch_hook(
        self,
        event_name: Literal["pre_run", "post_run", "pre_step", "post_step", "on_step_failure"],
        **kwargs: object,
    ) -> None:
        """Dispatch a hook to all registered hook functions."""
        await _dispatch_hook(self.hooks, event_name, **kwargs)

    def update_pipeline_result(
        self,
        result: PipelineResult[ContextT],
        step_result: StepResult,
    ) -> None:
        """Update the pipeline result with a step result."""
        replaced = False
        try:
            # Avoid duplicating synthesized "no terminal outcome" failures when padding
            # histories in multiple layers (ExecutionManager and run_session).
            if result.step_history:
                last = result.step_history[-1]
                if (
                    getattr(last, "name", None) == getattr(step_result, "name", None)
                    and getattr(last, "feedback", None) == getattr(step_result, "feedback", None)
                    and getattr(last, "success", True) is False
                ):
                    result.step_history[-1] = step_result
                    # Refresh cost/token aggregates for replacement
                    result.total_cost_usd -= getattr(last, "cost_usd", 0.0)
                    result.total_tokens -= getattr(last, "token_counts", 0)
                    replaced = True
        except Exception:
            replaced = False
        # Append the step result first unless we replaced the last entry
        if not replaced:
            result.step_history.append(step_result)

        # Base accumulation from the step itself
        result.total_cost_usd += step_result.cost_usd
        result.total_tokens += step_result.token_counts

        # If this step returned a nested PipelineResult (e.g., runner.as_step composition),
        # proactively aggregate its usage so top-level summaries remain correct.
        # This is a generic aggregation rule, not step-specific logic.
        try:
            nested = getattr(step_result, "output", None)
            # Prefer strict type check for clarity; fall back to duck-typing when needed
            is_nested_pr = isinstance(nested, PipelineResult)
            has_totals = (
                not is_nested_pr
                and nested is not None
                and hasattr(nested, "total_cost_usd")
                and hasattr(nested, "total_tokens")
                and hasattr(nested, "step_history")
            )
            if is_nested_pr or has_totals:
                try:
                    nested_cost = float(getattr(nested, "total_cost_usd", 0.0) or 0.0)
                except Exception:
                    nested_cost = 0.0
                try:
                    nested_tokens = int(getattr(nested, "total_tokens", 0) or 0)
                except Exception:
                    nested_tokens = 0

                # Heuristic: only add nested usage when the parent step hasn't already
                # accounted for it (common when using runner.as_step). If the parent
                # already aggregated, its own counts will be >= nested totals.
                add_cost = max(0.0, nested_cost - float(step_result.cost_usd or 0.0))
                add_tokens = max(0, nested_tokens - int(step_result.token_counts or 0))

                if add_cost > 0.0:
                    result.total_cost_usd += add_cost
                    # Reflect aggregation on the step result for consistent reporting
                    try:
                        step_result.cost_usd = float(step_result.cost_usd or 0.0) + add_cost
                    except Exception:
                        pass
                if add_tokens > 0:
                    result.total_tokens += add_tokens
                    try:
                        step_result.token_counts = int(step_result.token_counts or 0) + add_tokens
                    except Exception:
                        pass
                # Mark idempotency once we've applied nested usage
                try:
                    if getattr(step_result, "metadata_", None) is None:
                        step_result.metadata_ = {}
                    step_result.metadata_["nested_usage_aggregated"] = True
                except Exception:
                    pass

                # When the step did not record sub-step history, attach it for introspection
                try:
                    if isinstance(step_result.step_history, list) and not step_result.step_history:
                        sub_hist = getattr(nested, "step_history", [])
                        if isinstance(sub_hist, list) and sub_hist:
                            # Shallow-copy to avoid aliasing nested objects across containers
                            step_result.step_history.extend(list(sub_hist))
                except Exception:
                    pass
        except Exception as agg_ex:
            # Never let aggregation best-effort break execution; emit a debug breadcrumb.
            try:
                telemetry.logfire.debug(
                    f"Nested usage aggregation skipped due to error: {agg_ex!r}"
                )
            except Exception:
                pass
