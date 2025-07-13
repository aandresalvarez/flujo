import pytest
from unittest.mock import AsyncMock, Mock

from flujo.application.runner import Flujo
from flujo.domain.dsl import Step
from flujo.testing.utils import StubAgent, gather_result


@pytest.mark.asyncio
async def test_concrete_value_passes() -> None:
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ok"])})
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is True
    assert history.output == "ok"


@pytest.mark.asyncio
async def test_mock_output_raises_type_error() -> None:
    class BadAgent:
        async def run(self, *_args, **_kwargs):
            return Mock()

    agent = BadAgent()
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = Flujo(step)
    with pytest.raises(TypeError, match="returned a Mock object"):
        await gather_result(runner, "in")


@pytest.mark.asyncio
async def test_nested_mock_not_caught() -> None:
    nested = Mock()

    class NestedAgent:
        async def run(self, *_args, **_kwargs):
            return {"data": nested}

    agent = NestedAgent()
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = Flujo(step)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.success is True
    assert history.output["data"] is nested


@pytest.mark.asyncio
async def test_pipeline_stops_on_mock() -> None:
    good_agent = StubAgent(["ok"])

    class BadAgent:
        def __init__(self):
            self.run = AsyncMock(return_value=Mock())

    bad_agent = BadAgent()
    final_agent = StubAgent(["end"])

    step1 = Step.model_validate({"name": "a", "agent": good_agent})
    step2 = Step.model_validate({"name": "b", "agent": bad_agent})
    step3 = Step.model_validate({"name": "c", "agent": final_agent})
    pipeline = step1 >> step2 >> step3
    runner = Flujo(pipeline)

    with pytest.raises(TypeError, match="returned a Mock object"):
        await gather_result(runner, "start")

    assert good_agent.call_count == 1
    assert bad_agent.run.call_count == 1
    assert final_agent.call_count == 0
