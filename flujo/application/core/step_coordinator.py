"""Step execution coordination with telemetry and hook management."""

from __future__ import annotations

from typing import Any, AsyncIterator, Generic, Optional, TypeVar, Literal

from flujo.domain.backends import ExecutionBackend, StepExecutionRequest
from flujo.domain.models import BaseModel, StepResult, PipelineContext, PipelineResult, UsageLimits
from flujo.domain.dsl import Step
from flujo.domain.resources import AppResources
from flujo.exceptions import (
    ContextInheritanceError,
    PipelineAbortSignal,
    PipelineContextInitializationError,
    PausedException,
    UsageLimitExceededError,
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
        step: Step[Any, Any],
        data: Any,
        context: Optional[ContextT],
        backend: Optional[ExecutionBackend] = None,  # ✅ NEW: Receive the backend to call.
        *,
        stream: bool = False,
        step_executor: Optional[Any] = None,  # Legacy parameter for backward compatibility
        usage_limits: Optional[UsageLimits] = None,  # ✅ NEW: Usage limits for step execution
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
        await self._dispatch_hook(
            "pre_step",
            step=step,
            step_input=data,
            context=context,
            resources=self.resources,
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
                            if isinstance(item, StepResult):
                                step_result = item
                                yield item  # Yield StepResult objects
                            else:
                                yield item
                    except TypeError:
                        # If that fails, try as regular async function
                        step_result = await step_executor(
                            step, data, context, self.resources, stream=stream
                        )
                        if isinstance(step_result, StepResult):
                            yield step_result
                elif backend is not None:
                    # New approach: call backend directly
                    if stream:
                        # For streaming, we need to collect chunks and yield them
                        chunks = []

                        async def on_chunk(chunk: Any) -> None:
                            chunks.append(chunk)

                        request = StepExecutionRequest(
                            step=step,
                            input_data=data,
                            context=context,
                            resources=self.resources,
                            stream=stream,
                            on_chunk=on_chunk,
                            usage_limits=usage_limits,
                        )

                        # Call the backend directly
                        step_result = await backend.execute_step(request)

                        # Yield chunks first, then result
                        for chunk in chunks:
                            yield chunk
                        yield step_result
                    else:
                        # Non-streaming case
                        request = StepExecutionRequest(
                            step=step,
                            input_data=data,
                            context=context,
                            resources=self.resources,
                            stream=stream,
                            usage_limits=usage_limits,
                        )

                        # Call the backend directly
                        step_result = await backend.execute_step(request)
                        yield step_result
                else:
                    raise ValueError("Either backend or step_executor must be provided")

            except PausedException as e:
                # Handle pause for human input
                if isinstance(context, PipelineContext):
                    context.scratchpad["status"] = "paused"
                    context.scratchpad["pause_message"] = str(e)
                    scratch = context.scratchpad
                    if "paused_step_input" not in scratch:
                        scratch["paused_step_input"] = data
                raise
            except UsageLimitExceededError:
                # Re-raise usage limit exceptions to be handled by ExecutionManager
                raise
            except PipelineContextInitializationError:
                # Re-raise context initialization errors to be handled by ExecutionManager
                raise
            except ContextInheritanceError:
                # Re-raise context inheritance errors to be handled by ExecutionManager
                raise

            # Update telemetry span with step metadata
            if step_result and step_result.metadata_:
                for key, value in step_result.metadata_.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception as e:
                        telemetry.logfire.error(f"Error setting span attribute: {e}")

        # Handle step success/failure
        if step_result:
            if step_result.success:
                await self._dispatch_hook(
                    "post_step",
                    step_result=step_result,
                    context=context,
                    resources=self.resources,
                )
            else:
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
                # Don't halt here - let the execution manager handle step failure
                pass

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
