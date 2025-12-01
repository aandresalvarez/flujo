import pytest

from flujo.application.core.step_result_pool import build_pooled_step_result
from flujo.domain.models import StepResult


def test_pooled_step_result_deep_copies_and_clears_state() -> None:
    pooled = build_pooled_step_result(
        name="a",
        success=True,
        output={"x": 1},
        attempts=2,
        latency_s=0.1,
        token_counts={"total": 5},
        cost_usd=0.2,
        feedback="ok",
        metadata={"k": "v"},
        step_history=[StepResult(name="child", success=True)],
    )

    assert pooled.name == "a"
    assert pooled.metadata_ == {"k": "v"}
    assert pooled.step_history and pooled.step_history[0].name == "child"

    pooled.metadata_["k"] = "mutated"
    pooled.step_history[0].name = "mutated_child"

    pooled2 = build_pooled_step_result(name="b", success=False)

    assert pooled2.name == "b"
    assert pooled2.metadata_ == {}
    assert pooled2.step_history == []


@pytest.mark.parametrize(
    "success,feedback",
    [
        (True, None),
        (False, "err"),
    ],
)
def test_pooled_step_result_sets_fields(success: bool, feedback: str | None) -> None:
    sr = build_pooled_step_result(name="step", success=success, feedback=feedback)
    assert sr.success is success
    assert sr.feedback == feedback
