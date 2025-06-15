import pytest
from pydantic import BaseModel

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.testing.utils import StubAgent


class Ctx(BaseModel):
    count: int = 0


class AddOneAgent:
    async def run(self, data: int, *, pipeline_context: Ctx | None = None) -> int:
        if pipeline_context:
            pipeline_context.count += 1
        return data + 1


@pytest.mark.asyncio
async def test_pipeline_runner_shared_context_flow() -> None:
    step1 = Step("a", AddOneAgent())
    step2 = Step("b", AddOneAgent())
    runner = Flujo(step1 >> step2, context_model=Ctx, initial_context_data={"count": 0})
    result = await runner.run_async(1)
    assert result.final_pipeline_context.count == 2
    assert result.step_history[-1].output == 3


@pytest.mark.asyncio
async def test_existing_agents_without_context() -> None:
    agent = StubAgent(["ok"])
    step = Step("s", agent)
    runner = Flujo(step)
    result = await runner.run_async("hi")
    assert result.step_history[0].output == "ok"
