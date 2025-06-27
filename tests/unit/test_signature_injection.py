import warnings
import pytest
from flujo.domain.models import BaseModel

from flujo import Flujo, step
from flujo.domain.resources import AppResources
from flujo.testing.utils import gather_result
from flujo.deprecation import _warned_locations


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
async def legacy(_: int, *, pipeline_context: Ctx) -> int:
    return pipeline_context.num


@pytest.mark.asyncio
async def test_context_injected() -> None:
    runner = Flujo(add, context_model=Ctx)
    result = await gather_result(runner, 1)
    assert result.final_pipeline_context.num == 1
    assert result.step_history[-1].output == 1


@pytest.mark.asyncio
async def test_resources_injected() -> None:
    runner = Flujo(res_step, context_model=Ctx, resources=MyRes(tag="X"))
    result = await gather_result(runner, 0)
    assert result.step_history[-1].output == "X"


@pytest.mark.asyncio
async def test_pipeline_context_deprecated_warning() -> None:
    _warned_locations.clear()
    runner = Flujo(legacy, context_model=Ctx)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        await gather_result(runner, 0)
    assert any(isinstance(w.message, DeprecationWarning) for w in rec)
