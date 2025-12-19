from __future__ import annotations

from typing import TypeGuard

from ....domain.models import Failure, StepOutcome, StepResult, Success


class OutcomeNormalizationError(TypeError):
    """Raised when normalize_outcome receives an unsupported value type."""

    def __init__(self, *, step_name: str, value: object) -> None:
        message = (
            f"Expected StepOutcome or StepResult for step '{step_name}', "
            f"received {type(value).__name__}"
        )
        super().__init__(message)


def is_step_outcome(value: object) -> TypeGuard[StepOutcome[StepResult]]:
    """Return True when value is a structured StepOutcome."""
    return isinstance(value, StepOutcome)


def is_step_result(value: object) -> TypeGuard[StepResult]:
    """Return True when value is a concrete StepResult."""
    return isinstance(value, StepResult)


def normalize_outcome(
    value: StepOutcome[StepResult] | StepResult,
    *,
    step_name: str,
) -> StepOutcome[StepResult]:
    """
    Convert StepResult into Success/Failure outcomes while preserving feedback.

    Args:
        value: Outcome or raw StepResult returned by policies/executors.
        step_name: Best-effort step name for error context.

    Returns:
        StepOutcome with Success/Failure semantics.

    Raises:
        TypeError: if the value cannot be normalized into a StepOutcome.
    """
    if is_step_outcome(value):
        return value
    if is_step_result(value):
        if value.success:
            return Success(step_result=value)
        return Failure(
            error=Exception(value.feedback or f"Step '{step_name}' failed"),
            feedback=value.feedback,
            step_result=value,
        )
    raise OutcomeNormalizationError(step_name=step_name, value=value)


__all__ = ["is_step_outcome", "is_step_result", "normalize_outcome"]
