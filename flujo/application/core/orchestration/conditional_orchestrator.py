"""Conditional step orchestration extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....domain.models import BaseModel

from ....infra import telemetry as _telemetry
from ....domain.models import StepOutcome, StepResult

if TYPE_CHECKING:  # pragma: no cover
    from ..types import ExecutionFrame


class ConditionalOrchestrator:
    """Runs conditional steps and emits telemetry."""

    async def execute(
        self,
        *,
        core: object,
        frame: "ExecutionFrame[BaseModel]",
    ) -> StepResult:
        from ..types import ExecutionFrame as _ExecutionFrame

        if not isinstance(frame, _ExecutionFrame):
            raise TypeError("ConditionalOrchestrator.execute expects an ExecutionFrame")

        step = frame.step
        with _telemetry.logfire.span(getattr(step, "name", "<unnamed>")) as _span:
            executor = getattr(core, "conditional_step_executor", None)
            execute_fn = getattr(executor, "execute", None)
            if not callable(execute_fn):
                raise TypeError("conditional_step_executor missing execute()")
            outcome_raw = await execute_fn(core, frame)
            if not isinstance(outcome_raw, (StepOutcome, StepResult)):
                raise TypeError(
                    "conditional_step_executor returned unsupported type "
                    f"{type(outcome_raw).__name__}"
                )
            outcome: StepOutcome[StepResult] | StepResult = outcome_raw
        unwrap_fn = getattr(core, "_unwrap_outcome_to_step_result", None)
        safe_step_name_fn = getattr(core, "_safe_step_name", None)
        if not callable(unwrap_fn) or not callable(safe_step_name_fn):
            raise TypeError("ExecutorCore missing outcome unwrap helpers")
        sr_raw = unwrap_fn(outcome, safe_step_name_fn(step))
        if not isinstance(sr_raw, StepResult):
            raise TypeError(
                f"ConditionalOrchestrator expected StepResult, got {type(sr_raw).__name__}"
            )
        sr = sr_raw
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
