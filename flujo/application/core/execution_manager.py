"""Main execution manager that orchestrates pipeline execution components."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Optional, TypeVar, Generic

from ...domain.dsl.pipeline import Pipeline
from ...domain.models import BaseModel, PipelineResult, StepResult

from ...infra import telemetry

from .state_manager import StateManager
from .usage_governor import UsageGovernor
from .step_coordinator import StepCoordinator
from .type_validator import TypeValidator

ContextT = TypeVar("ContextT", bound=BaseModel)


class ExecutionManager(Generic[ContextT]):
    """Main execution manager that orchestrates all execution components."""

    def __init__(
        self,
        pipeline: Pipeline[Any, Any],
        *,
        state_manager: Optional[StateManager[ContextT]] = None,
        usage_governor: Optional[UsageGovernor[ContextT]] = None,
        step_coordinator: Optional[StepCoordinator[ContextT]] = None,
        type_validator: Optional[TypeValidator] = None,
    ) -> None:
        self.pipeline = pipeline
        self.state_manager = state_manager or StateManager()
        self.usage_governor = usage_governor or UsageGovernor()
        self.step_coordinator = step_coordinator or StepCoordinator()
        self.type_validator = type_validator or TypeValidator()

    async def execute_steps(
        self,
        start_idx: int,
        data: Any,
        context: Optional[ContextT],
        result: PipelineResult[ContextT],
        *,
        stream_last: bool = False,
        run_id: str | None = None,
        state_created_at: datetime | None = None,
        step_executor: Any,  # StepExecutor type
    ) -> AsyncIterator[Any]:
        """Execute pipeline steps with simplified, coordinated logic.

        This is the main execution loop that coordinates all components:
        - Step execution via StepCoordinator
        - State persistence via StateManager
        - Usage limit checking via UsageGovernor
        - Type validation via TypeValidator

        Args:
            start_idx: Index of first step to execute
            data: Input data for first step
            context: Pipeline context
            result: Pipeline result to populate
            stream_last: Whether to stream final step output
            run_id: Workflow run ID for state persistence
            state_created_at: When state was created
            step_executor: Function to execute individual steps

        Yields:
            Streaming output chunks or step results
        """
        for idx, step in enumerate(self.pipeline.steps[start_idx:], start=start_idx):
            # Execute step with coordination
            step_result = None
            async for item in self.step_coordinator.execute_step(
                step,
                data,
                context,
                stream=stream_last and idx == len(self.pipeline.steps) - 1,
                step_executor=step_executor,
            ):
                if isinstance(item, StepResult):
                    step_result = item
                else:
                    yield item

            if step_result is None:
                continue

            # Update pipeline result
            self.step_coordinator.update_pipeline_result(result, step_result)

            # Check usage limits - this may raise UsageLimitExceededError
            with telemetry.logfire.span(step.name) as span:
                self.usage_governor.check_usage_limits(result, span)
                self.usage_governor.update_telemetry_span(span, result)

            # Persist state if needed
            if step_result.success and run_id is not None:
                await self.state_manager.persist_workflow_state(
                    run_id=run_id,
                    context=context,
                    current_step_index=idx + 1,
                    last_step_output=step_result.output,
                    status="running",
                    state_created_at=state_created_at,
                )

            # Stop on step failure
            if not step_result.success:
                break

            # Validate type compatibility with next step - this may raise TypeMismatchError
            if idx < len(self.pipeline.steps) - 1:
                next_step = self.pipeline.steps[idx + 1]
                self.type_validator.validate_step_output(step, step_result.output, next_step)

            # Pass output to next step
            data = step_result.output

    def set_final_context(
        self,
        result: PipelineResult[ContextT],
        context: Optional[ContextT],
    ) -> None:
        """Set the final context in the pipeline result."""
        if context is not None:
            result.final_pipeline_context = context

    async def persist_final_state(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        result: PipelineResult[ContextT],
        start_idx: int,
        state_created_at: datetime | None,
        final_status: str,
    ) -> None:
        """Persist final workflow state."""
        if run_id is None:
            return

        last_step_output = result.step_history[-1].output if result.step_history else None

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=start_idx + len(result.step_history),
            last_step_output=last_step_output,
            status=final_status,
            state_created_at=state_created_at,
        )
