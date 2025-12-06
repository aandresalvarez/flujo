"""Conditional step orchestration extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...infra import telemetry as _telemetry
from ...domain.models import StepOutcome, StepResult

if TYPE_CHECKING:  # pragma: no cover
    from .executor_core import ExecutorCore
    from .types import ExecutionFrame


class ConditionalOrchestrator:
    """Runs conditional steps and emits telemetry."""

    async def execute(
        self,
        *,
        core: "ExecutorCore[Any]",
        frame: "ExecutionFrame[Any]",
    ) -> StepResult:
        from .types import ExecutionFrame as _ExecutionFrame

        if not isinstance(frame, _ExecutionFrame):
            raise TypeError("ConditionalOrchestrator.execute expects an ExecutionFrame")

        step = frame.step
        with _telemetry.logfire.span(getattr(step, "name", "<unnamed>")) as _span:
            outcome: StepOutcome[StepResult] = await core.conditional_step_executor.execute(
                core, frame
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
                    # Ensure warning is logged with "No branch" prefix for test compatibility
                    warn_msg = fb if "No branch" in fb or "no branch" in fb else f"No branch: {fb}"
                    _telemetry.logfire.warn(warn_msg)
                elif fb:
                    _telemetry.logfire.error(fb)
        except Exception:
            pass
        return sr
