import asyncio
from typing import Any

from flujo.domain.models import BaseModel
import pytest

from flujo.domain import Step, Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    nums: list[int]


class DoubleAgent:
    async def run(self, data: int, **kwargs) -> int:
        await asyncio.sleep(0)
        return data * 2


@pytest.mark.asyncio
async def test_map_over_sequential() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "double", "agent": DoubleAgent()}))
    mapper = Step.map_over("mapper", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=Ctx)
    result = await gather_result(runner, None, initial_context_data={"nums": [1, 2, 3]})
    assert result.step_history[-1].output == [2, 4, 6]


class SleepAgent:
    async def run(self, data: int, **kwargs) -> int:
        await asyncio.sleep(0.01)
        return data


@pytest.mark.asyncio
async def test_map_over_parallel() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "sleep", "agent": SleepAgent()}))
    mapper = Step.map_over("mapper_par", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=Ctx)
    result = await gather_result(runner, None, initial_context_data={"nums": [0, 1, 2, 3]})
    assert result.step_history[-1].output == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_map_over_empty() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "double", "agent": DoubleAgent()}))
    mapper = Step.map_over("mapper_empty", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=Ctx)
    result = await gather_result(runner, None, initial_context_data={"nums": []})
    assert result.step_history[-1].output == []


class LooseCtx(BaseModel):
    nums: Any


@pytest.mark.asyncio
async def test_map_over_invalid_input() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "double", "agent": DoubleAgent()}))
    mapper = Step.map_over("mapper_invalid", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=LooseCtx)
    result = await gather_result(runner, None, initial_context_data={"nums": 42})
    assert not result.step_history[-1].success
    assert "iterable" in result.step_history[-1].feedback


@pytest.mark.asyncio
async def test_map_over_reusable_after_empty() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "double", "agent": DoubleAgent()}))
    mapper = Step.map_over("mapper_reuse", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=Ctx)
    first = await gather_result(runner, None, initial_context_data={"nums": []})
    assert first.step_history[-1].output == []
    second = await gather_result(
        runner,
        None,
        initial_context_data={"nums": [3, 4]},
    )
    assert second.step_history[-1].output == [6, 8]


@pytest.mark.asyncio
async def test_map_over_concurrent_runs() -> None:
    body = Pipeline.from_step(Step.model_validate({"name": "double", "agent": DoubleAgent()}))
    mapper = Step.map_over("mapper_concurrent", body, iterable_input="nums")
    runner = create_test_flujo(mapper, context_model=Ctx)

    async def run_one(vals: list[int]) -> Any:
        return await gather_result(
            runner,
            None,
            initial_context_data={"nums": vals},
        )

    r1, r2 = await asyncio.gather(run_one([1, 2]), run_one([5, 6]))
    assert r1.step_history[-1].output == [2, 4]
    assert r2.step_history[-1].output == [10, 12]
