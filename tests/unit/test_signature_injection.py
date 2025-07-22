import pytest
from flujo.domain.models import BaseModel

from flujo import step
from flujo.domain.resources import AppResources
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    num: int = 0


class MyRes(AppResources):
    tag: str = "ok"


@step
async def add(x: int, *, context: Ctx) -> int:
    context.num += x
    return context.num


@step
async def res_step(_: int, *, resources: MyRes) -> str:
    return resources.tag


@step
async def legacy(_: int, *, context: Ctx) -> int:
    return context.num


@pytest.mark.asyncio
async def test_context_injected() -> None:
    runner = create_test_flujo(add, context_model=Ctx)
    result = await gather_result(runner, 1)
    assert result.final_pipeline_context.num == 1
    assert result.step_history[-1].output == 1


@pytest.mark.asyncio
async def test_resources_injected() -> None:
    runner = create_test_flujo(res_step, context_model=Ctx, resources=MyRes(tag="X"))
    result = await gather_result(runner, 0)
    assert result.step_history[-1].output == "X"


@pytest.mark.asyncio
async def test_legacy_context_works() -> None:
    runner = create_test_flujo(legacy, context_model=Ctx)
    result = await gather_result(runner, 0)
    assert result.step_history[-1].output == 0
