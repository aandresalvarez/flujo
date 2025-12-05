"""Human-in-the-loop step orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...domain.models import (
    StepResult,
    StepOutcome,
    Paused,
    Success,
    Failure,
)
from ...exceptions import PausedException

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore
    from .types import ExecutionFrame


class HitlOrchestrator:
    """Handles HITL execution semantics and pause propagation."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        frame: "ExecutionFrame[Any]",
    ) -> StepResult:
        step = frame.step
        context = frame.context
        outcome: StepOutcome[StepResult] = await core.hitl_step_executor.execute(core, frame)
        if isinstance(outcome, Paused):
            try:
                if context is not None and hasattr(context, "scratchpad"):
                    scratch = getattr(context, "scratchpad")
                    if isinstance(scratch, dict):
                        scratch["status"] = "paused"
                        scratch["pause_message"] = outcome.message
            except Exception:
                pass
            raise PausedException(outcome.message)
        if isinstance(outcome, Success):
            return outcome.step_result
        if isinstance(outcome, Failure):
            return core._unwrap_outcome_to_step_result(outcome, getattr(step, "name", "<hitl>"))
        return StepResult(
            name=getattr(step, "name", "<hitl>"), success=False, feedback="Unsupported HITL outcome"
        )
