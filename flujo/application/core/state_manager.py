"""State management for workflow persistence and resumption."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TypeVar, Generic

from ...domain.models import BaseModel, PipelineContext
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
    ) -> tuple[Optional[ContextT], Any, int, Optional[datetime], Optional[str], Optional[str]]:
        """Load workflow state from persistence backend.

        Returns:
            Tuple of (context, last_step_output, current_step_index, created_at, pipeline_name, pipeline_version)
        """
        if self.state_backend is None or not run_id:
            return None, None, 0, None, None, None

        loaded = await self.state_backend.load_state(run_id)
        if loaded is None:
            return None, None, 0, None, None, None

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

        return (
            context,
            wf_state.last_step_output,
            wf_state.current_step_index,
            wf_state.created_at,
            wf_state.pipeline_name,
            wf_state.pipeline_version,
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

        state_data = {
            "run_id": run_id,
            "pipeline_id": getattr(context, "pipeline_id", "unknown"),
            "pipeline_name": getattr(context, "pipeline_name", "unknown"),
            "pipeline_version": getattr(context, "pipeline_version", "latest"),
            "current_step_index": current_step_index,
            "pipeline_context": context.model_dump() if context is not None else None,
            "last_step_output": last_step_output,
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
