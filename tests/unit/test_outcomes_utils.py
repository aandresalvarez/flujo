from flujo.domain.models import StepResult, Failure, StepOutcome
from flujo.domain.outcomes import to_outcome, unwrap


def test_to_outcome_and_unwrap_roundtrip_success():
    sr = StepResult(name="s", success=True, output={"x": 1})
    oc = to_outcome(sr)
    assert isinstance(oc, StepOutcome)
    un = unwrap(oc, step_name="s")
    assert un == sr


def test_to_outcome_and_unwrap_roundtrip_failure_with_feedback():
    sr = StepResult(name="s", success=False, feedback="bad")
    oc = to_outcome(sr)
    assert isinstance(oc, Failure)
    un = unwrap(oc, step_name="s")
    assert un == sr
