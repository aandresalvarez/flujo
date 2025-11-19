import importlib
import pytest
from flujo import Flujo, Step
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo


def test_core_imports() -> None:
    assert importlib.import_module("flujo")
    assert Flujo
    assert Step


@pytest.mark.asyncio
async def test_basic_pipeline_runs() -> None:
    step = Step.model_validate({"name": "s1", "agent": StubAgent(["ok"])})
    result = await gather_result(create_test_flujo(step), "hi")
    assert result.step_history[-1].output == "ok"
