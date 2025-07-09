"""Main execution manager that orchestrates pipeline execution components."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator, Optional, TypeVar, Generic, Dict

from ...domain.dsl.pipeline import Pipeline
from ...domain.models import BaseModel, PipelineResult, StepResult
from ...domain.ir import PipelineIR
from ...registry import CallableRegistry
from ...testing.utils import get_default_agent_registry

from ...exceptions import (
    PausedException,
    PipelineAbortSignal,
    UsageLimitExceededError,
    PipelineContextInitializationError,
)
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
        pipeline: Pipeline[Any, Any] | PipelineIR[Any, Any],
        *,
        state_manager: Optional[StateManager[ContextT]] = None,
        usage_governor: Optional[UsageGovernor[ContextT]] = None,
        step_coordinator: Optional[StepCoordinator[ContextT]] = None,
        type_validator: Optional[TypeValidator] = None,
        agent_registry: Optional[Dict[str, Any]] = None,
        callable_registry: Optional[CallableRegistry] = None,
        plugin_registry: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.pipeline_ir = pipeline if isinstance(pipeline, PipelineIR) else None

        # Auto-populate agent registry if not provided
        if agent_registry is None:
            try:
                agent_registry = get_default_agent_registry()
            except Exception:
                # Fallback to empty dict if discovery fails
                agent_registry = {}

        self.agent_registry = agent_registry
        self.callable_registry = callable_registry or CallableRegistry()
        self.state_manager = state_manager or StateManager()
        self.usage_governor = usage_governor or UsageGovernor()
        self.step_coordinator = step_coordinator or StepCoordinator()
        self.type_validator = type_validator or TypeValidator()
        self.plugin_registry = plugin_registry or {}

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
        # Determine which steps to iterate over
        steps_to_execute: list[Any]
        if self.pipeline_ir is not None:
            # Use IR and rehydrate steps
            steps_to_execute_ir: list[Any] = []
            for step_ir in self.pipeline_ir.steps[start_idx:]:
                step = Pipeline.from_model(
                    PipelineIR(steps=[step_ir]),
                    self.agent_registry,
                    self.callable_registry,
                    self.plugin_registry,
                ).steps[0]
                steps_to_execute_ir.append(step)
            steps_to_execute = steps_to_execute_ir
        else:
            # Use original pipeline steps
            steps_to_execute = list(self.pipeline.steps[start_idx:])

        for idx, step in enumerate(steps_to_execute, start=start_idx):
            step_result = None
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

                    # Persist state if needed
                    if step_result and step_result.success and run_id is not None:
                        await self.state_manager.persist_workflow_state(
                            run_id=run_id,
                            context=context,
                            current_step_index=idx + 1,
                            last_step_output=step_result.output,
                            status="running",
                            state_created_at=state_created_at,
                        )

                    # Validate type compatibility with next step - this may raise TypeMismatchError
                    total_steps = (
                        len(self.pipeline_ir.steps)
                        if self.pipeline_ir
                        else len(self.pipeline.steps)
                    )
                    if step_result and idx < total_steps - 1:
                        if self.pipeline_ir is not None:
                            next_step_ir = self.pipeline_ir.steps[idx + 1]
                            next_step_ir_instance: Any = Pipeline.from_model(
                                PipelineIR(steps=[next_step_ir]),
                                self.agent_registry,
                                self.callable_registry,
                                self.plugin_registry,
                            ).steps[0]
                        else:
                            next_step_pipeline: Any = self.pipeline.steps[idx + 1]
                        self.type_validator.validate_step_output(
                            step,
                            step_result.output,
                            next_step_ir_instance
                            if self.pipeline_ir is not None
                            else next_step_pipeline,
                        )

                    # Pass output to next step
                    if step_result:
                        data = step_result.output

                except PipelineAbortSignal:
                    # Update pipeline result before aborting
                    if step_result is not None:
                        self.step_coordinator.update_pipeline_result(result, step_result)
                    self.set_final_context(result, context)
                    yield result
                    return
                except PausedException as e:
                    # Handle pause by updating context and returning current result
                    if context is not None:
                        if hasattr(context, "scratchpad"):
                            context.scratchpad["status"] = "paused"
                            context.scratchpad["pause_message"] = str(e)
                    self.set_final_context(result, context)
                    yield result
                    return
                except UsageLimitExceededError:
                    # Re-raise usage limit errors to be handled by the runner
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
                    # Do not raise here; let finally block handle

            finally:
                if step_result is not None and (
                    not result.step_history or result.step_history[-1] is not step_result
                ):
                    self.step_coordinator.update_pipeline_result(result, step_result)

                    # Check usage limits after step result is added to pipeline result
                    with telemetry.logfire.span(step.name) as span:
                        self.usage_governor.check_usage_limits(result, span)
                        self.usage_governor.update_telemetry_span(span, result)

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
