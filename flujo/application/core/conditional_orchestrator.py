"""Conditional step orchestration extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ...infra import telemetry as _telemetry
from ...domain.models import PipelineResult, StepOutcome, StepResult, UsageLimits

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore


class ConditionalOrchestrator:
    """Runs conditional steps and emits telemetry."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        fallback_depth: int = 0,
    ) -> StepResult:
        with _telemetry.logfire.span(getattr(step, "name", "<unnamed>")) as _span:
            outcome: StepOutcome[StepResult] = await core.conditional_step_executor.execute(
                core, step, data, context, resources, limits, context_setter, fallback_depth
            )
        sr = core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(step))
        try:
            md = getattr(sr, "metadata_", None) or {}
            branch_key = md.get("executed_branch_key")
            if branch_key is not None:
                _telemetry.logfire.info(f"Condition evaluated to branch key '{branch_key}'")
                _telemetry.logfire.info(f"Executing branch for key '{branch_key}'")
                try:
                    _span.set_attribute("executed_branch_key", branch_key)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if not getattr(sr, "success", False):
                fb = (sr.feedback or "") if hasattr(sr, "feedback") else ""
                if "no branch" in fb.lower():
                    _telemetry.logfire.warn(fb)
                elif fb:
                    _telemetry.logfire.error(fb)
        except Exception:
            pass
        return sr
