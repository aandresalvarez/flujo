"""Human-in-the-loop step orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....domain.models import BaseModel

from ....domain.models import (
    StepResult,
    StepOutcome,
    Paused,
    Success,
    Failure,
)
from ....exceptions import PausedException

if TYPE_CHECKING:  # pragma: no cover
    from ..types import ExecutionFrame


class HitlOrchestrator:
    """Handles HITL execution semantics and pause propagation."""

    async def execute(
        self,
        *,
        core: object,
        frame: "ExecutionFrame[BaseModel]",
    ) -> StepResult:
        step = frame.step
        context = frame.context
        executor = getattr(core, "hitl_step_executor", None)
        execute_fn = getattr(executor, "execute", None)
        if not callable(execute_fn):
            raise TypeError("hitl_step_executor missing execute()")
        outcome_raw = await execute_fn(core, frame)
        if isinstance(outcome_raw, StepResult):
            return outcome_raw
        if not isinstance(outcome_raw, StepOutcome):
            raise TypeError(
                f"hitl_step_executor returned unsupported type {type(outcome_raw).__name__}"
            )
        if isinstance(outcome_raw, Paused):
            try:
                if context is not None:
                    # Use typed fields instead of scratchpad for status and pause_message
                    if hasattr(context, "status"):
                        context.status = "paused"
                    if hasattr(context, "pause_message"):
                        context.pause_message = outcome_raw.message
            except Exception:
                pass
            raise PausedException(outcome_raw.message)
        if isinstance(outcome_raw, Success):
            return outcome_raw.step_result
        if isinstance(outcome_raw, Failure):
            if outcome_raw.step_result is not None:
                return outcome_raw.step_result
            return StepResult(
                name=getattr(step, "name", "<hitl>"),
                success=False,
                feedback=outcome_raw.feedback or "HITL failed",
            )
        return StepResult(
            name=getattr(step, "name", "<hitl>"), success=False, feedback="Unsupported HITL outcome"
        )
