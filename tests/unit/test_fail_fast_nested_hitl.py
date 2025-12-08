import pytest

from flujo.domain.dsl import Pipeline, Step
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.exceptions import ConfigurationError


def test_fail_fast_nested_hitl():
    """Nested HITL inside conditional inside loop should fail validation."""

    hitl = Step.human_in_the_loop(name="hitl", message_for_user="Provide input")

    cond = ConditionalStep(
        name="cond",
        condition_callable=lambda *_: "yes",
        branches={"yes": Pipeline.from_step(hitl)},
        default_branch_pipeline=Pipeline.from_step(Step.from_callable(lambda x: x, name="noop")),
    )

    loop = LoopStep(
        name="loop",
        loop_body_pipeline=Pipeline.from_step(cond),
        exit_condition_callable=lambda *_: True,
        max_retries=1,
    )

    pipeline = Pipeline.from_step(loop)

    with pytest.raises(ConfigurationError):
        pipeline.validate_graph(raise_on_error=True)
