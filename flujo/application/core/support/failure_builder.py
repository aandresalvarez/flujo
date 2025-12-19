"""Failure outcome construction helpers."""

from __future__ import annotations

from typing import Callable, TypeVar
from ....domain.models import BaseModel

from ....domain.models import Failure, StepResult
from ..types import ExecutionFrame


ContextT = TypeVar("ContextT", bound=BaseModel)


def build_failure_outcome(
    *,
    step: object,
    frame: ExecutionFrame[ContextT],
    exc: Exception,
    called_with_frame: bool,
    safe_step_name: Callable[[object], str],
) -> Failure[StepResult]:
    """Create a Failure outcome with branch context when appropriate."""
    step_name = safe_step_name(step)
    err_type = type(exc).__name__ if hasattr(exc, "__class__") else "Exception"
    include_branch_context = called_with_frame and getattr(step, "updates_context", False)
    branch_ctx = frame.context if include_branch_context else None
    failed_sr = StepResult(
        name=step_name,
        output=None,
        success=False,
        attempts=1,
        latency_s=0.0,
        token_counts=0,
        cost_usd=0.0,
        feedback=f"{err_type}: {exc}",
        branch_context=branch_ctx,
        metadata_={"error_type": err_type, "called_with_frame": called_with_frame},
    )
    return Failure(error=exc, feedback=failed_sr.feedback, step_result=failed_sr)
