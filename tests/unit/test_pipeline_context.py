import pytest
from pydantic import BaseModel

from flujo.application.flujo_engine import Flujo
from flujo.domain import Step
from flujo.testing.utils import gather_result
from flujo.exceptions import PipelineContextInitializationError
from flujo.domain.plugins import PluginOutcome


class Ctx(BaseModel):
    num: int = 0


class CaptureAgent:
    def __init__(self):
        self.seen = None

    async def run(self, data: str, *, pipeline_context: Ctx | None = None) -> str:
        self.seen = pipeline_context
        return data


class IncAgent:
    async def run(self, data: str, *, pipeline_context: Ctx | None = None) -> str:
        assert pipeline_context is not None
        pipeline_context.num += 1
        return data


class ReadAgent:
    async def run(self, data: str, *, pipeline_context: Ctx | None = None) -> int:
        assert pipeline_context is not None
        return pipeline_context.num


class ContextPlugin:
    def __init__(self):
        self.ctx = None

    async def validate(self, data: dict, *, pipeline_context: Ctx | None = None) -> PluginOutcome:
        self.ctx = pipeline_context
        return PluginOutcome(success=True)


class StrictPlugin:
    async def validate(self, data: dict) -> PluginOutcome:
        return PluginOutcome(success=True)


class KwargsPlugin:
    def __init__(self):
        self.kwargs = None

    async def validate(self, data: dict, **kwargs) -> PluginOutcome:
        self.kwargs = kwargs
        return PluginOutcome(success=True)


@pytest.mark.asyncio
async def test_context_initialization_and_access() -> None:
    agent = CaptureAgent()
    step = Step("s", agent)
    runner = Flujo(step, context_model=Ctx, initial_context_data={"num": 1})
    result = await gather_result(runner, "in")
    assert isinstance(agent.seen, Ctx)
    assert result.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_context_initialization_failure() -> None:
    runner = Flujo(
        Step("s", CaptureAgent()), context_model=Ctx, initial_context_data={"num": "bad"}
    )
    with pytest.raises(PipelineContextInitializationError):
        await gather_result(runner, "in")


@pytest.mark.asyncio
async def test_context_mutation_between_steps() -> None:
    pipeline = Step("inc", IncAgent()) >> Step("read", ReadAgent())
    runner = Flujo(pipeline, context_model=Ctx)
    result = await gather_result(runner, "x")
    assert result.step_history[-1].output == 1
    assert result.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_context_isolated_per_run() -> None:
    step = Step("inc", IncAgent())
    runner = Flujo(step, context_model=Ctx)
    r1 = await gather_result(runner, "a")
    r2 = await gather_result(runner, "b")
    assert r1.final_pipeline_context.num == 1
    assert r2.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_plugin_receives_context_and_strict_plugin_errors() -> None:
    ctx_plugin = ContextPlugin()
    kwargs_plugin = KwargsPlugin()
    strict_plugin = StrictPlugin()
    step = Step(
        "s", CaptureAgent(), plugins=[(ctx_plugin, 0), (kwargs_plugin, 0), (strict_plugin, 0)]
    )
    runner = Flujo(step, context_model=Ctx)
    with pytest.raises(TypeError):
        await gather_result(runner, "in")
    # Context plugin ran before the TypeError
    assert isinstance(ctx_plugin.ctx, Ctx)
    assert kwargs_plugin.kwargs.get("pipeline_context") == ctx_plugin.ctx
