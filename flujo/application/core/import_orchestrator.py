"""Import step orchestration extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ...domain.dsl.import_step import ImportStep
from ...domain.models import PipelineResult, StepOutcome, StepResult, UsageLimits
from .executor_helpers import make_execution_frame

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class ImportOrchestrator:
    """Executes ImportStep via the configured import_step_executor."""

    def __init__(self, executor: Optional[Any]) -> None:
        self._executor = executor

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: ImportStep,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        frame: Optional[Any] = None,
    ) -> StepOutcome[StepResult]:
        if frame is None:
            frame = make_execution_frame(
                core,
                step,
                data,
                context,
                resources,
                limits,
                context_setter=context_setter,
                stream=False,
                on_chunk=None,
                fallback_depth=0,
                result=None,
                quota=core._get_current_quota() if hasattr(core, "_get_current_quota") else None,
            )
        if self._executor is None:
            result = await core._policy_default_step(frame)
            return result
        executor_result: StepOutcome[StepResult] = await self._executor.execute(core, frame)
        # ImportStep semantics: if the child run returned no output but the child context
        # carries scratchpad updates, merge them back to the parent context to preserve
        # side effects (e.g., computed SQL or state machine transitions).
        if (
            isinstance(executor_result, StepOutcome)
            and hasattr(executor_result, "step_result")
            and executor_result.step_result is not None
        ):
            sr = executor_result.step_result
            try:
                merge_child_ctx = (
                    sr.output is None
                    and getattr(step, "outputs", None) is None
                    and context is not None
                    and hasattr(context, "scratchpad")
                )
                if merge_child_ctx:
                    child_ctx = getattr(sr, "branch_context", None)
                    if child_ctx is not None and hasattr(child_ctx, "scratchpad"):
                        child_sp = getattr(child_ctx, "scratchpad", None)
                        parent_sp = getattr(context, "scratchpad", None)
                        if isinstance(child_sp, dict) and isinstance(parent_sp, dict):
                            parent_sp.update(child_sp)
                            setattr(context, "scratchpad", parent_sp)
            except Exception:
                pass
        return executor_result
