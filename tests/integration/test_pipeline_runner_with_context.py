import asyncio
import pytest
from flujo.domain.models import BaseModel

from flujo.domain import Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.models import PipelineResult
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    count: int = 0


class AddOneAgent:
    async def run(self, data: int, *, context: Ctx | None = None) -> int:
        if context:
            context.count += 1
        return data + 1


@pytest.mark.asyncio
async def test_pipeline_runner_shared_context_flow() -> None:
    step1 = Step.model_validate({"name": "a", "agent": AddOneAgent()})
    step2 = Step.model_validate({"name": "b", "agent": AddOneAgent()})
    runner = create_test_flujo(step1 >> step2, context_model=Ctx, initial_context_data={"count": 0})
    result = await gather_result(runner, 1)
    assert result.final_pipeline_context.count == 2
    assert result.step_history[-1].output == 3


@pytest.mark.asyncio
async def test_existing_agents_without_context() -> None:
    agent = StubAgent(["ok"])
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = create_test_flujo(step)
    result = await gather_result(runner, "hi")
    assert result.step_history[0].output == "ok"


class _TestContext(BaseModel):
    counter: int = 0


class IncrementAgent:
    async def run(self, data: str, *, context: _TestContext | None = None) -> dict:
        if context:
            context.counter += 1
        return {"counter": context.counter if context else 0}


@pytest.mark.asyncio
async def test_concurrent_runs_with_typed_context_are_isolated() -> None:
    step = Step.model_validate({"name": "inc", "agent": IncrementAgent(), "updates_context": True})
    runner = create_test_flujo(step, context_model=_TestContext)

    async def run_one() -> PipelineResult:
        return await gather_result(runner, "input")

    result1, result2 = await asyncio.gather(run_one(), run_one())

    assert result1.final_pipeline_context.counter == 1
    assert result2.final_pipeline_context.counter == 1
