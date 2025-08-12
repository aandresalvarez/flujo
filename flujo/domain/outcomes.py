from __future__ import annotations


from .models import StepOutcome, StepResult, Success, Failure, Paused


def to_outcome(step_result: StepResult) -> StepOutcome[StepResult]:
    if isinstance(step_result, StepOutcome):
        return step_result
    return (
        Success(step_result=step_result)
        if step_result.success
        else Failure(
            error=Exception(step_result.feedback or "step failed"),
            feedback=step_result.feedback,
            step_result=step_result,
        )
    )


def unwrap(
    outcome_or_result: StepOutcome[StepResult] | StepResult, *, step_name: str = "unknown"
) -> StepResult:
    if isinstance(outcome_or_result, StepResult):
        return outcome_or_result
    if isinstance(outcome_or_result, Success):
        return outcome_or_result.step_result
    if isinstance(outcome_or_result, Failure):
        if outcome_or_result.step_result is not None:
            return outcome_or_result.step_result
        return StepResult(
            name=step_name,
            output=None,
            success=False,
            feedback=outcome_or_result.feedback
            or (str(outcome_or_result.error) if outcome_or_result.error is not None else None),
        )
    if isinstance(outcome_or_result, Paused):
        from ..exceptions import PausedException

        raise PausedException(outcome_or_result.message)
    return StepResult(
        name=step_name,
        output=None,
        success=False,
        feedback=f"Unsupported outcome type: {type(outcome_or_result).__name__}",
    )
