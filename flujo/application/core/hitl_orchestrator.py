"""Human-in-the-loop step orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Awaitable

from ...domain.models import (
    PipelineResult,
    StepResult,
    StepOutcome,
    UsageLimits,
    Paused,
    Success,
    Failure,
)
from ...exceptions import PausedException

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class HitlOrchestrator:
    """Handles HITL execution semantics and pause propagation."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        cache_key: Optional[str] = None,
        fallback_depth: int = 0,
    ) -> StepResult:
        outcome: StepOutcome[StepResult] = await core.hitl_step_executor.execute(
            core, step, data, context, resources, limits, context_setter
        )
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
