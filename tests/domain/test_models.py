import json
from typing import Any


from flujo.domain.models import (
    StepResult,
    Success,
    Failure,
    Paused,
    Aborted,
    Chunk,
)


def _make_step_result(**kwargs: Any) -> StepResult:
    base = dict(
        name="test-step",
        output={"x": 1},
        success=True,
        attempts=1,
        latency_s=0.01,
        token_counts=10,
        cost_usd=0.001,
        feedback=None,
        branch_context=None,
        metadata_={},
        step_history=[],
    )
    base.update(kwargs)
    return StepResult(**base)


def test_success_outcome_instantiation_and_serialization() -> None:
    sr = _make_step_result()
    outcome = Success(step_result=sr)
    dumped = outcome.model_dump_json()
    assert isinstance(dumped, str)
    payload = json.loads(dumped)
    assert payload["step_result"]["name"] == "test-step"
    assert payload["step_result"]["success"] is True


def test_failure_outcome_instantiation_and_serialization() -> None:
    partial = _make_step_result(success=False, feedback="bad")
    err = ValueError("boom")
    outcome = Failure(error=str(err), feedback="bad", step_result=partial)
    dumped = outcome.model_dump_json()
    payload = json.loads(dumped)
    assert payload["feedback"] == "bad"
    assert payload["step_result"]["success"] is False


def test_paused_outcome_instantiation_and_serialization() -> None:
    outcome = Paused(message="awaiting input", state_token={"k": "v"})
    dumped = outcome.model_dump_json()
    payload = json.loads(dumped)
    assert payload["message"] == "awaiting input"
    assert payload["state_token"]["k"] == "v"


def test_aborted_outcome_instantiation_and_serialization() -> None:
    outcome = Aborted(reason="circuit breaker tripped")
    dumped = outcome.model_dump_json()
    payload = json.loads(dumped)
    assert payload["reason"] == "circuit breaker tripped"


def test_chunk_outcome_instantiation_and_serialization() -> None:
    outcome = Chunk(data="partial", step_name="step-1")
    dumped = outcome.model_dump_json()
    payload = json.loads(dumped)
    assert payload["data"] == "partial"
    assert payload["step_name"] == "step-1"
