import asyncio
import pytest
from pydantic import BaseModel
from flujo.domain import Step
from flujo.testing.utils import gather_result
from flujo.application.flujo_engine import Flujo


class Ctx(BaseModel):
    val: int = 0


class AddAgent:
    def __init__(self, inc: int) -> None:
        self.inc = inc

    async def run(self, data: int, *, context: Ctx | None = None) -> int:
        if context is not None:
            context.val += self.inc
        await asyncio.sleep(0)
        return data + self.inc


@pytest.mark.asyncio
async def test_parallel_step_context_isolation() -> None:
    branches = {
        "a": Step("a", AddAgent(1)),
        "b": Step("b", AddAgent(2)),
    }
    parallel = Step.parallel("par", branches)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, 0)
    step_result = result.step_history[-1]
    assert step_result.output == {"a": 1, "b": 2}
    assert result.final_pipeline_context.val == 0


@pytest.mark.asyncio
async def test_parallel_step_result_structure() -> None:
    branches = {
        "x": Step("x", AddAgent(3)),
        "y": Step("y", AddAgent(4)),
    }
    parallel = Step.parallel("par_out", branches)
    runner = Flujo(parallel, context_model=Ctx)
    result = await gather_result(runner, 1)
    step_result = result.step_history[-1]
    assert isinstance(step_result.output, dict)
    assert set(step_result.output.keys()) == {"x", "y"}
    assert step_result.success is True
