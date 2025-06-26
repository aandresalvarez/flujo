import asyncio
from pydantic import BaseModel
import pytest

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step, Pipeline
from flujo.testing.utils import gather_result


class Ctx(BaseModel):
    nums: list[int]


class DoubleAgent:
    async def run(self, data: int, **kwargs) -> int:
        await asyncio.sleep(0)
        return data * 2


@pytest.mark.asyncio
async def test_map_over_sequential() -> None:
    body = Pipeline.from_step(Step("double", DoubleAgent()))
    mapper = Step.map_over("mapper", body, iterable_input="nums")
    runner = Flujo(mapper, context_model=Ctx)
    result = await gather_result(runner, None, initial_context_data={"nums": [1, 2, 3]})
    assert result.step_history[-1].output == [2, 4, 6]


class SleepAgent:
    async def run(self, data: int, **kwargs) -> int:
        await asyncio.sleep(0.01)
        return data


@pytest.mark.asyncio
async def test_map_over_parallel() -> None:
    body = Pipeline.from_step(Step("sleep", SleepAgent()))
    mapper = Step.map_over("mapper_par", body, iterable_input="nums", max_concurrency=2)
    runner = Flujo(mapper, context_model=Ctx)
    result = await gather_result(runner, None, initial_context_data={"nums": [0, 1, 2, 3]})
    assert result.step_history[-1].output == [0, 1, 2, 3]
