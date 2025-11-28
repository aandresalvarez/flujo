from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

ContextT = TypeVar("ContextT")


class ExecutionFinalizationMixin(Generic[ContextT]):
    """Finalization hooks for ExecutionManager (backward-compatible stub)."""

    async def persist_final_state(
        self,
        result: Any,
        context: Optional[ContextT],
        *,
        run_id: Optional[str] = None,
        state_manager: Any = None,
        pipeline: Any = None,
        **_: Any,
    ) -> None:  # pragma: no cover - stub
        # Persist via state manager when provided; otherwise best-effort set final context.
        if state_manager is None:
            state_manager = getattr(self, "state_manager", None)
        if run_id is None and state_manager is not None:
            try:
                run_id = state_manager.get_run_id_from_context(context)  # type: ignore[attr-defined]
            except Exception:
                pass
        step_history = getattr(result, "step_history", None) or []
        start_idx = _.get("start_idx", 0) or 0
        current_idx = start_idx + len(step_history)
        last_output = None
        try:
            if step_history:
                last_output = getattr(step_history[-1], "output", None)
        except Exception:
            last_output = None
        final_status = _.get("final_status", "completed")
        state_created_at = _.get("state_created_at", None)
        if state_manager is not None and run_id is not None:
            try:
                if hasattr(state_manager, "persist_workflow_state"):
                    await state_manager.persist_workflow_state(
                        run_id=run_id,
                        context=context,
                        current_step_index=current_idx,
                        last_step_output=last_output,
                        status=final_status,
                        state_created_at=state_created_at,
                        step_history=step_history,
                    )
                try:
                    if hasattr(state_manager, "record_run_end"):
                        await state_manager.record_run_end(run_id=run_id, result=result)
                except Exception:
                    pass
                self.set_final_context(result, context)
                return
            except Exception:
                pass
        self.set_final_context(result, context)

    def set_final_context(
        self, result: Any, context: Optional[ContextT]
    ) -> None:  # pragma: no cover - stub
        # Attach the final context to the result when possible.
        try:
            setattr(result, "final_context", context)
        except Exception:
            pass
        try:
            # Align with PipelineResult field used in tests
            setattr(result, "final_pipeline_context", context)
        except Exception:
            pass
