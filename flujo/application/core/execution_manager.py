"""Main execution manager that orchestrates pipeline execution components."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Optional, TypeVar, Generic

from ...domain.dsl.pipeline import Pipeline
from ...domain.dsl.loop import LoopStep
from ...domain.models import (
    BaseModel,
    PipelineResult,
    StepResult,
)
from ...infra import telemetry

from ...exceptions import (
    PausedException,
    PipelineAbortSignal,
    UsageLimitExceededError,
    PipelineContextInitializationError,
)
from .state_manager import StateManager
from .usage_governor import UsageGovernor
from .step_coordinator import StepCoordinator
from .type_validator import TypeValidator

ContextT = TypeVar("ContextT", bound=BaseModel)


class ExecutionManager(Generic[ContextT]):
    """Main execution manager that orchestrates all execution components.

    This class coordinates step execution, state management, usage governance,
    and type validation for pipeline execution. It can be configured to run
    inside loop steps to provide proper context isolation and state management.
    """

    def __init__(
        self,
        pipeline: Pipeline[Any, Any],
        *,
        state_manager: Optional[StateManager[ContextT]] = None,
        usage_governor: Optional[UsageGovernor[ContextT]] = None,
        step_coordinator: Optional[StepCoordinator[ContextT]] = None,
        type_validator: Optional[TypeValidator] = None,
        inside_loop_step: bool = False,
    ) -> None:
        """Initialize the execution manager.

        Args:
            pipeline: The pipeline to execute
            state_manager: Optional state manager for persistence
            usage_governor: Optional usage governor for limits
            step_coordinator: Optional step coordinator for execution
            type_validator: Optional type validator for compatibility
            inside_loop_step: Whether this manager is running inside a loop step.
                When True, enables proper context isolation and state management
                for loop iterations to prevent unintended side effects between
                iterations and ensure each iteration operates independently.
        """
        self.pipeline = pipeline
        self.state_manager = state_manager or StateManager()
        self.usage_governor = usage_governor or UsageGovernor()
        self.step_coordinator = step_coordinator or StepCoordinator()
        self.type_validator = type_validator or TypeValidator()
        self.inside_loop_step = inside_loop_step  # Track if we're inside a loop step

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
            step_result = None
            should_add_step_result = (
                True  # Default to adding step result unless explicitly prevented
            )
            try:
                try:
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

                    # Validate type compatibility with next step - this may raise TypeMismatchError
                    if step_result and idx < len(self.pipeline.steps) - 1:
                        next_step = self.pipeline.steps[idx + 1]
                        self.type_validator.validate_step_output(
                            step, step_result.output, next_step
                        )

                    # Pass output to next step
                    if step_result:
                        data = step_result.output

                    # CRITICAL FIX: Check usage limits immediately after step completes
                    # This ensures we stop before the next step starts
                    if step_result is not None:
                        # Only perform usage limit checks if limits are configured
                        if self.usage_governor.usage_limits is not None:
                            # Calculate potential total cost only when needed
                            potential_total_cost = result.total_cost_usd + step_result.cost_usd

                            # Create a new step history list for the temporary result (do not mutate the original).
                            # This ensures the exception contains the correct state at the moment of the limit breach.
                            # Performance note: List concatenation creates a new list, but this is necessary for correctness
                            # and exception traceability. For extremely large pipelines, consider on-disk step history.
                            from flujo.domain.models import PipelineResult

                            temp_result: PipelineResult[ContextT] = PipelineResult(
                                step_history=result.step_history + [step_result],
                                total_cost_usd=potential_total_cost,
                                final_pipeline_context=result.final_pipeline_context,
                                trace_tree=result.trace_tree,
                            )
                            with telemetry.logfire.span(f"usage_check_{step.name}") as span:
                                self.usage_governor.check_usage_limits(temp_result, span)
                                self.usage_governor.update_telemetry_span(span, temp_result)
                        # If we reach here, the usage check passed, so keep should_add_step_result = True

                except PipelineAbortSignal:
                    # Update pipeline result before aborting
                    # Add current step result to pipeline result before yielding
                    if step_result is not None:
                        self.step_coordinator.update_pipeline_result(result, step_result)
                    should_add_step_result = False  # Prevent duplicate addition in finally block
                    self.set_final_context(result, context)
                    yield result
                    return
                except PausedException as e:
                    # Handle pause by updating context and returning current result
                    if context is not None:
                        if hasattr(context, "scratchpad"):
                            context.scratchpad["status"] = "paused"
                            context.scratchpad["pause_message"] = str(e)
                    # Add current step result to pipeline result before yielding
                    if step_result is not None:
                        self.step_coordinator.update_pipeline_result(result, step_result)
                    should_add_step_result = False  # Prevent duplicate addition in finally block
                    self.set_final_context(result, context)
                    yield result
                    return
                except UsageLimitExceededError:
                    # Add the breaching step to the PipelineResult for consistency before re-raising the exception
                    if step_result is not None:
                        self.step_coordinator.update_pipeline_result(result, step_result)
                    should_add_step_result = False
                    raise
                except PipelineContextInitializationError as e:
                    # Convert to ContextInheritanceError if appropriate
                    from ...exceptions import ContextInheritanceError
                    from ..context_manager import _extract_missing_fields

                    # Attach the exception itself to a dummy StepResult for finally block
                    step_result = StepResult(
                        name=step.name,
                        output=None,
                        success=False,
                        attempts=0,
                        feedback=str(e),
                        metadata_={
                            "_context_init_cause": e.__cause__
                            if e.__cause__ is not None
                            else e.__context__
                        },
                    )
                    should_add_step_result = True
                    # Do not raise here; let finally block handle

            finally:
                if self._should_add_step_result_to_pipeline(
                    step_result, should_add_step_result, result
                ):
                    assert (
                        step_result is not None
                    )  # Type checker: we know it's not None from the helper method
                    self.step_coordinator.update_pipeline_result(result, step_result)

                    # Persist state after every step for robust crash recovery
                    # Skip persistence for loop steps to avoid context serialization issues
                    if (
                        run_id is not None
                        and not isinstance(step, LoopStep)
                        and not self.inside_loop_step
                    ):
                        await self.state_manager.persist_workflow_state(
                            run_id=run_id,
                            context=context,
                            current_step_index=idx + 1,
                            last_step_output=step_result.output,
                            status="running",
                            state_created_at=state_created_at,
                            step_history=result.step_history,
                        )
                        # Always record step result for observability
                        await self.state_manager.record_step_result(run_id, step_result, idx)
                    elif run_id is not None and (
                        isinstance(step, LoopStep) or self.inside_loop_step
                    ):
                        # Always record step result for observability, skip state persistence
                        await self.state_manager.record_step_result(run_id, step_result, idx)
                        telemetry.logfire.debug(
                            f"Skipped state persistence for {'LoopStep' if isinstance(step, LoopStep) else 'inner step'} '{step.name}' to avoid context serialization issues"
                        )

                    # If the step failed due to context inheritance, propagate the error
                    if step_result is not None and not step_result.success and step_result.feedback:
                        if (
                            "Failed to inherit context" in step_result.feedback
                            or "Missing required fields" in step_result.feedback
                        ):
                            from ...exceptions import ContextInheritanceError
                            from ..context_manager import _extract_missing_fields

                            cause = None
                            if (
                                step_result.metadata_
                                and "_context_init_cause" in step_result.metadata_
                            ):
                                cause = step_result.metadata_["_context_init_cause"]
                            missing_fields = _extract_missing_fields(cause)
                            if not missing_fields and step_result.feedback:
                                import re

                                match = re.search(
                                    r"Missing required fields: ([\w, ]+)", step_result.feedback
                                )
                                if match:
                                    missing_fields = [f.strip() for f in match.group(1).split(",")]
                            raise ContextInheritanceError(
                                missing_fields=missing_fields,
                                parent_context_keys=[],
                                child_model_name="Unknown",
                            )

            if step_result is None:
                continue

            # Stop on step failure
            if not step_result.success:
                break

    def _should_add_step_result_to_pipeline(
        self,
        step_result: Optional[StepResult],
        should_add_step_result: bool,
        result: PipelineResult[ContextT],
    ) -> bool:
        """Determine if a step result should be added to the pipeline result.

        Args:
            step_result: The step result to potentially add
            should_add_step_result: Flag indicating if the step result should be added
            result: The current pipeline result

        Returns:
            True if the step result should be added to the pipeline result
        """
        return (
            step_result is not None
            and should_add_step_result
            and (not result.step_history or result.step_history[-1] is not step_result)
        )

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
            step_history=result.step_history,
        )
        await self.state_manager.record_run_end(run_id, result)
