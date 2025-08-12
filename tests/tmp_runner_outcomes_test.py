import pytest

from flujo.application.runner import Flujo
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import Success


class _EchoAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        return f"ok:{payload}"


@pytest.mark.asyncio
async def test_tmp_runner_outcomes_non_streaming():
    step = Step(name="echo", agent=_EchoAgent())
    pipe = Pipeline.from_step(step)
    f = Flujo(pipe)
    last = None
    async for item in f.run_outcomes_async("hi"):
        last = item
    assert isinstance(last, Success)
    assert last.step_result.name == "echo"
