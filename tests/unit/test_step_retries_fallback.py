import pytest
from flujo.domain import Step, step
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome

@pytest.mark.asyncio
async def test_step_decorator_retries_sets_config() -> None:
    @step(retries=3)
    async def do(x: int) -> int:
        return x

    assert do.config.max_retries == 3

@pytest.mark.asyncio
async def test_step_from_callable_fallback_argument() -> None:
    async def primary(x: str) -> str:
        return x

    fb = Step.from_callable(lambda x: "fb")
    step_instance = Step.from_callable(primary, fallback=fb)
    assert step_instance.fallback_step is fb
