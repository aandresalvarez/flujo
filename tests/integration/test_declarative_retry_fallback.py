import pytest
from flujo.domain import Step, step
from flujo.testing.utils import DummyPlugin, gather_result
from flujo.application.runner import Flujo
from flujo.domain.plugins import PluginOutcome

@pytest.mark.asyncio
async def test_decorator_retries_and_fallback_executes() -> None:
    async def fb(x: str) -> str:
        return "ok"

    fallback_step = Step.from_callable(fb, name="fb")

    @step(retries=3, fallback=fallback_step)
    async def primary(x: str) -> str:
        return "bad"

    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")] * 3)
    primary.plugins.append((plugin, 0))

    runner = Flujo(primary)
    result = await gather_result(runner, "data")
    sr = result.step_history[0]
    assert sr.success is True
    assert sr.output == "ok"
    assert sr.metadata_["fallback_triggered"] is True
    assert plugin.call_count == 3
