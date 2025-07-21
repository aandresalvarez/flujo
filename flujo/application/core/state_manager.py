"""State management for workflow persistence and resumption."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional, TypeVar, Generic

from ...domain.models import BaseModel, PipelineContext, PipelineResult, StepResult
from ...state import StateBackend, WorkflowState

ContextT = TypeVar("ContextT", bound=BaseModel)


class StateManager(Generic[ContextT]):
    """Manages workflow state persistence and loading for durable execution."""

    def __init__(self, state_backend: Optional[StateBackend] = None) -> None:
        self.state_backend = state_backend

    async def load_workflow_state(
        self,
        run_id: str,
        context_model: Optional[type[ContextT]] = None,
    ) -> tuple[
        Optional[ContextT],
        Any,
        int,
        Optional[datetime],
        Optional[str],
        Optional[str],
        list[StepResult],
    ]:
        """Load workflow state from persistence backend.

        Returns:
            Tuple of (context, last_step_output, current_step_index, created_at, pipeline_name, pipeline_version, step_history)
        """
        if self.state_backend is None or not run_id:
            return None, None, 0, None, None, None, []

        loaded = await self.state_backend.load_state(run_id)
        if loaded is None:
            return None, None, 0, None, None, None, []

        wf_state = WorkflowState.model_validate(loaded)

        # Reconstruct context from persisted state
        context: Optional[ContextT] = None
        if wf_state.pipeline_context is not None:
            if context_model is not None:
                context = context_model.model_validate(wf_state.pipeline_context)
            else:
                context = PipelineContext.model_validate(wf_state.pipeline_context)  # type: ignore

            # Restore pipeline metadata from state
            if context is not None and hasattr(context, "pipeline_name"):
                context.pipeline_name = wf_state.pipeline_name
            if context is not None and hasattr(context, "pipeline_version"):
                context.pipeline_version = wf_state.pipeline_version

        # Reconstruct step history from persisted state
        step_history: list[StepResult] = []
        for step_data in wf_state.step_history:
            try:
                step_result = StepResult.model_validate(step_data)
                step_history.append(step_result)
            except Exception:
                # Skip invalid step data to avoid breaking resumption
                continue

        return (
            context,
            wf_state.last_step_output,
            wf_state.current_step_index,
            wf_state.created_at,
            wf_state.pipeline_name,
            wf_state.pipeline_version,
            step_history,
        )

    async def persist_workflow_state(
        self,
        *,
        run_id: str | None,
        context: Optional[ContextT],
        current_step_index: int,
        last_step_output: Any | None,
        status: str,
        state_created_at: datetime | None = None,
        step_history: Optional[list[StepResult]] = None,
    ) -> None:
        """Persist current workflow state to backend."""
        if self.state_backend is None or run_id is None:
            return

        # Calculate execution time if we have creation timestamp
        execution_time_ms = None
        if state_created_at is not None:
            execution_time_ms = int((datetime.now() - state_created_at).total_seconds() * 1000)

        # Estimate memory usage (rough approximation)
        memory_usage_mb = None
        if context is not None:
            try:
                import sys

                memory_usage_mb = sys.getsizeof(context) / (1024 * 1024)
            except Exception:
                pass

        # Optimize serialization by only dumping context when necessary
        pipeline_context = None
        if context is not None:
            try:
                pipeline_context = context.model_dump()
            except Exception:
                # Fallback to basic serialization if model_dump fails
                pipeline_context = {"error": "Failed to serialize context"}

        # Serialize step history
        serialized_step_history = []
        if step_history is not None:
            for step_result in step_history:
                try:
                    serialized_step_history.append(step_result.model_dump())
                except Exception:
                    # Skip invalid step results to avoid breaking persistence
                    continue

        state_data = {
            "run_id": run_id,
            "pipeline_id": getattr(context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(context, "pipeline_version", "latest"),
            "current_step_index": current_step_index,
            "pipeline_context": pipeline_context,
            "last_step_output": last_step_output,
            "step_history": serialized_step_history,
            "status": status,
            "created_at": state_created_at or datetime.now(),
            "updated_at": datetime.now(),
            "total_steps": getattr(context, "total_steps", 0),
            "error_message": getattr(context, "error_message", None),
            "execution_time_ms": execution_time_ms,
            "memory_usage_mb": memory_usage_mb,
        }

        await self.state_backend.save_state(run_id, state_data)

    def get_run_id_from_context(self, context: Optional[ContextT]) -> str | None:
        """Extract run_id from context if available."""
        if context is None:
            return None
        return getattr(context, "run_id", None)

    async def delete_workflow_state(self, run_id: str | None) -> None:
        """Delete workflow state from backend."""
        if self.state_backend is None or run_id is None:
            return

        await self.state_backend.delete_state(run_id)

    # ----------------------- New persistence helpers -----------------------

    async def record_run_start(
        self,
        run_id: str,
        pipeline_id: str,
        pipeline_name: str,
        pipeline_version: str,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        if self.state_backend is None:
            return
        try:
            # Use provided timestamps or generate defaults
            now = datetime.utcnow().isoformat()
            created_at = created_at or now
            updated_at = updated_at or now

            await self.state_backend.save_run_start(
                {
                    "run_id": run_id,
                    "pipeline_id": pipeline_id,
                    "pipeline_name": pipeline_name,
                    "pipeline_version": pipeline_version,
                    "status": "running",
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        except NotImplementedError:
            pass

    async def record_step_result(
        self, run_id: str, step_result: StepResult, step_index: int
    ) -> None:
        if self.state_backend is None:
            return
        try:
            await self.state_backend.save_step_result(
                {
                    "step_run_id": f"{run_id}:{step_index}",
                    "run_id": run_id,
                    "step_name": step_result.name,
                    "step_index": step_index,
                    "status": "completed" if step_result.success else "failed",
                    "start_time": datetime.utcnow(),
                    "end_time": datetime.utcnow(),
                    "duration_ms": int(step_result.latency_s * 1000),
                    "cost": step_result.cost_usd,
                    "tokens": step_result.token_counts,
                    "input": None,
                    "output": step_result.output,
                    "error": step_result.feedback if not step_result.success else None,
                }
            )
        except NotImplementedError:
            pass

    async def record_run_end(self, run_id: str, result: PipelineResult[ContextT]) -> None:
        if self.state_backend is None:
            return
        try:
            await self.state_backend.save_run_end(
                run_id,
                {
                    "status": "completed"
                    if all(s.success for s in result.step_history)
                    else "failed",
                    "end_time": datetime.utcnow(),
                    "total_cost": result.total_cost_usd,
                    "final_context": result.final_pipeline_context.model_dump()
                    if result.final_pipeline_context
                    else None,
                },
            )

            # Save trace tree if available
            if result.trace_tree is not None:
                try:
                    # Convert trace tree to dict format for JSON serialization
                    trace_dict = self._convert_trace_to_dict(result.trace_tree)
                    await self.state_backend.save_trace(run_id, trace_dict)
                except Exception as e:
                    # Log error and save error trace for auditability
                    from ...infra import telemetry

                    telemetry.logfire.error(f"Failed to save trace for run {run_id}: {e}")

                    # Save sanitized error trace for auditability
                    # Sanitize error message to prevent sensitive data leakage
                    error_message = str(e)
                    # Truncate and sanitize error message to prevent sensitive data leakage
                    sanitized_error = (
                        error_message[:100] + "..." if len(error_message) > 100 else error_message
                    )
                    # Remove potential sensitive patterns
                    import re

                    sanitized_error = re.sub(
                        r"(password|secret|key|token|api_key)\s*[:=]\s*\S+",
                        r"\1=***",
                        sanitized_error,
                        flags=re.IGNORECASE,
                    )

                    error_trace = {
                        "span_id": f"error_{run_id}",
                        "name": "trace_save_error",
                        "start_time": datetime.now().timestamp(),
                        "end_time": datetime.now().timestamp(),
                        "parent_span_id": None,
                        "attributes": {
                            "error_type": type(e).__name__,
                            "error_summary": f"Trace serialization failed: {sanitized_error}",
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
                        },
                        "children": [],
                        "status": "error",
                    }

                    try:
                        await self.state_backend.save_trace(run_id, error_trace)
                        telemetry.logfire.info(
                            f"Saved error trace for run {run_id} after trace save failure"
                        )
                    except Exception as save_error:
                        telemetry.logfire.error(
                            f"Failed to save error trace for run {run_id}: {save_error}"
                        )
        except NotImplementedError:
            pass

    def _convert_trace_to_dict(self, trace_tree: Any) -> Dict[str, Any]:
        """Convert trace tree to dictionary format for JSON serialization."""
        if hasattr(trace_tree, "__dict__"):
            # Handle Span objects
            trace_dict: Dict[str, Any] = {
                "span_id": getattr(trace_tree, "span_id", "unknown"),
                "name": getattr(trace_tree, "name", "unknown"),
                "start_time": getattr(trace_tree, "start_time", 0.0),
                "end_time": getattr(trace_tree, "end_time", 0.0),
                "parent_span_id": getattr(trace_tree, "parent_span_id", None),
                "attributes": getattr(trace_tree, "attributes", {}),
                "children": [],
                "status": getattr(trace_tree, "status", "unknown"),
            }
            # Convert children recursively
            children = getattr(trace_tree, "children", [])
            for child in children:
                if isinstance(trace_dict["children"], list):
                    trace_dict["children"].append(self._convert_trace_to_dict(child))
            return trace_dict
        elif isinstance(trace_tree, dict):
            # Already a dict, just ensure children are converted
            if "children" in trace_tree:
                converted_children = []
                for child in trace_tree["children"]:
                    converted_children.append(self._convert_trace_to_dict(child))
                trace_tree["children"] = converted_children
            return trace_tree
        else:
            # Raise exception for truly invalid trace trees to trigger error handling
            raise ValueError(f"Unknown trace tree type: {type(trace_tree)}")
