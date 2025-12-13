from __future__ import annotations

import pytest

from flujo.application.core.type_guards import (
    is_step_outcome,
    is_step_result,
    normalize_outcome,
)
from flujo.domain.models import Failure, StepResult, Success


def test_normalize_outcome_wraps_success() -> None:
    result = StepResult(name="s", success=True, output="ok")

    normalized = normalize_outcome(result, step_name="s")

    assert isinstance(normalized, Success)
    assert normalized.step_result is result


def test_normalize_outcome_wraps_failure_with_feedback() -> None:
    result = StepResult(name="s", success=False, feedback="bad")

    normalized = normalize_outcome(result, step_name="s")

    assert isinstance(normalized, Failure)
    assert normalized.feedback == "bad"
    assert normalized.step_result is result


def test_normalize_outcome_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        normalize_outcome(object(), step_name="unknown")


def test_type_guards_reflect_outcome_and_result() -> None:
    outcome = Success(step_result=StepResult(name="s", success=True))

    assert is_step_outcome(outcome)
    assert is_step_result(outcome.step_result)
