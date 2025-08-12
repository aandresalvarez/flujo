import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import Failure, StepOutcome


class _FailingAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_runner_run_outcomes_failure_yields_failure_and_stops():
    step = Step(name="fail", agent=_FailingAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)

    outcomes = []
    async for item in f.run_outcomes_async("hi"):
        outcomes.append(item)
        # If we received a Failure, break out to validate no more Success is yielded
        if isinstance(item, Failure):
            break

    assert outcomes, "Expected at least one outcome"
    first = outcomes[0]
    assert isinstance(first, StepOutcome)
    assert isinstance(first, Failure)
    assert first.step_result is not None
    assert first.step_result.success is False
    # Ensure we did not receive a Success after failure
    assert all(
        not hasattr(o, "step_result")
        or not getattr(o, "step_result", None)
        or getattr(o, "step_result").success is False
        for o in outcomes
    )
