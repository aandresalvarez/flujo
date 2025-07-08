"""Step execution coordination with telemetry and hook management."""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional, TypeVar, Generic

from ...domain.dsl.step import Step
from ...domain.models import BaseModel, PipelineResult, StepResult, PipelineContext
from ...domain.resources import AppResources
from ...domain.types import HookCallable
from typing import Literal
from ...exceptions import PausedException, PipelineAbortSignal, PipelineContextInitializationError
from ...infra import telemetry
from ..core.hook_dispatcher import _dispatch_hook as _dispatch_hook_impl

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
        *,
        stream: bool = False,
        step_executor: Any,  # StepExecutor type
    ) -> AsyncIterator[Any]:
        """Execute a single step with telemetry and hook management.

        Args:
            step: The step to execute
            data: Input data for the step
            context: Pipeline context
            stream: Whether to stream output
            step_executor: Function to execute the step

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
                async for item in step_executor(step, data, context, self.resources, stream=stream):
                    if isinstance(item, StepResult):
                        step_result = item
                    else:
                        yield item
            except PausedException as e:
                # Handle pause for human input
                if isinstance(context, PipelineContext):
                    context.scratchpad["status"] = "paused"
                    context.scratchpad["pause_message"] = str(e)
                    scratch = context.scratchpad
                    if "paused_step_input" not in scratch:
                        scratch["paused_step_input"] = data
                raise
            except PipelineContextInitializationError:
                # Re-raise context initialization errors to be handled by ExecutionManager
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
                telemetry.logfire.warn(f"Step '{step.name}' failed. Halting pipeline execution.")

            yield step_result

    async def _dispatch_hook(
        self,
        event_name: Literal["pre_run", "post_run", "pre_step", "post_step", "on_step_failure"],
        **kwargs: Any,
    ) -> None:
        """Dispatch hooks for the given event."""
        try:
            await _dispatch_hook_impl(self.hooks, event_name, **kwargs)
        except PipelineAbortSignal:
            # Re-raise PipelineAbortSignal so it propagates up to ExecutionManager
            raise

    def update_pipeline_result(
        self,
        result: PipelineResult[ContextT],
        step_result: StepResult,
    ) -> None:
        """Update pipeline result with step result."""
        result.step_history.append(step_result)
        result.total_cost_usd += step_result.cost_usd
