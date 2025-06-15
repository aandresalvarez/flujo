import pytest
from flujo.domain import Step, Pipeline, ConditionalStep


def test_conditional_step_init_validation() -> None:
    with pytest.raises(ValueError):
        ConditionalStep(
            name="cond",
            condition_callable=lambda *_: "a",
            branches={},
        )


def test_step_factory_branch_on() -> None:
    branches = {"a": Pipeline.from_step(Step("a"))}
    step = Step.branch_on(
        name="branch",
        condition_callable=lambda *_: "a",
        branches=branches,
    )
    assert isinstance(step, ConditionalStep)
    assert "a" in step.branches
