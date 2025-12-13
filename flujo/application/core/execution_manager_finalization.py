from __future__ import annotations

from typing import Generic, TypeVar

ContextT = TypeVar("ContextT")


class ExecutionFinalizationMixin(Generic[ContextT]):
    """Finalization hooks for ExecutionManager (backward-compatible stub)."""

    async def persist_final_state(
        self,
        result: object,
        context: ContextT | None,
        *,
        run_id: str | None = None,
        state_manager: object | None = None,
        pipeline: object | None = None,
        **kwargs: object,
    ) -> None:  # pragma: no cover - stub
        # Persist via state manager when provided; otherwise best-effort set final context.
        if state_manager is None:
            state_manager = getattr(self, "state_manager", None)
        if run_id is None and state_manager is not None:
            try:
                get_run_id = getattr(state_manager, "get_run_id_from_context", None)
                if callable(get_run_id):
                    run_id = get_run_id(context)
            except Exception:
                pass
        step_history = getattr(result, "step_history", None) or []
        start_idx = kwargs.get("start_idx", 0) or 0
        current_idx = start_idx + len(step_history)
        last_output = None
        try:
            if step_history:
                last_output = getattr(step_history[-1], "output", None)
        except Exception:
            last_output = None
        final_status = kwargs.get("final_status", "completed")
        state_created_at = kwargs.get("state_created_at", None)
        if state_manager is not None and run_id is not None:
            try:
                persist_fn = getattr(state_manager, "persist_workflow_state", None)
                if callable(persist_fn):
                    await persist_fn(
                        run_id=run_id,
                        context=context,
                        current_step_index=current_idx,
                        last_step_output=last_output,
                        status=final_status,
                        state_created_at=state_created_at,
                        step_history=step_history,
                    )
                try:
                    record_end = getattr(state_manager, "record_run_end", None)
                    if callable(record_end):
                        await record_end(run_id=run_id, result=result)
                except Exception:
                    pass
                self.set_final_context(result, context)
                return
            except Exception:
                pass
        self.set_final_context(result, context)

    def set_final_context(
        self, result: object, context: ContextT | None
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
