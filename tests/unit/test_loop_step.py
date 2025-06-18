import pytest

from flujo.domain import Step, Pipeline, LoopStep


def test_loop_step_init_validation() -> None:
    with pytest.raises(ValueError):
        LoopStep(
            name="loop",
            loop_body_pipeline=Pipeline.from_step(Step("a")),
            exit_condition_callable=lambda *_: True,
            max_loops=0,
        )


def test_step_factory_loop_until() -> None:
    body = Pipeline.from_step(Step("a"))
    step = Step.loop_until(
        name="loop", loop_body_pipeline=body, exit_condition_callable=lambda *_: True
    )
    assert isinstance(step, LoopStep)
    assert step.max_loops == 5
