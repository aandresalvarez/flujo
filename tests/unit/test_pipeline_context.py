import pytest
from typing import Optional
from flujo.domain.models import BaseModel
from flujo.domain import Step
from flujo import step
from flujo.testing.utils import gather_result
from flujo.exceptions import PipelineContextInitializationError
from flujo.domain.plugins import PluginOutcome
from tests.conftest import create_test_flujo


class Ctx(BaseModel):
    num: int = 0


class CaptureAgent:
    def __init__(self):
        self.seen = None

    async def run(self, data: str, *, context: Optional[Ctx] = None) -> str:
        self.seen = context
        return data


class IncAgent:
    async def run(self, data: str, *, context: Optional[Ctx] = None) -> str:
        assert context is not None
        context.num += 1
        return data


class ReadAgent:
    async def run(self, data: str, *, context: Optional[Ctx] = None) -> int:
        assert context is not None
        return context.num


class ContextPlugin:
    def __init__(self):
        self.ctx = None

    async def validate(self, data: dict, *, context: Optional[Ctx] = None) -> PluginOutcome:
        self.ctx = context
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
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = create_test_flujo(step, context_model=Ctx, initial_context_data={"num": 1})
    result = await gather_result(runner, "in")
    assert isinstance(agent.seen, Ctx)
    assert result.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_context_initialization_failure() -> None:
    runner = create_test_flujo(
        Step.model_validate({"name": "s", "agent": CaptureAgent()}),
        context_model=Ctx,
        initial_context_data={"num": "bad"},
    )
    with pytest.raises(PipelineContextInitializationError):
        await gather_result(runner, "in")


@pytest.mark.asyncio
async def test_context_mutation_between_steps() -> None:
    pipeline = Step.model_validate({"name": "inc", "agent": IncAgent()}) >> Step.model_validate(
        {"name": "read", "agent": ReadAgent()}
    )
    runner = create_test_flujo(pipeline, context_model=Ctx)
    result = await gather_result(runner, "x")
    assert result.step_history[-1].output == 1
    assert result.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_context_isolated_per_run() -> None:
    step = Step.model_validate({"name": "inc", "agent": IncAgent()})
    runner = create_test_flujo(step, context_model=Ctx)
    r1 = await gather_result(runner, "a")
    r2 = await gather_result(runner, "b")
    assert r1.final_pipeline_context.num == 1
    assert r2.final_pipeline_context.num == 1


@pytest.mark.asyncio
async def test_plugin_receives_context_and_strict_plugin_errors() -> None:
    ctx_plugin = ContextPlugin()
    kwargs_plugin = KwargsPlugin()
    strict_plugin = StrictPlugin()
    step = Step.model_validate(
        {
            "name": "s",
            "agent": CaptureAgent(),
            "plugins": [(ctx_plugin, 0), (kwargs_plugin, 0), (strict_plugin, 0)],
        }
    )
    runner = create_test_flujo(step, context_model=Ctx)
    # This should run without error, as the engine is smart enough not to pass
    # context to a plugin that doesn't accept it.
    result = await gather_result(runner, "in")

    # Verify that the run succeeded and plugins received context correctly.
    assert result.step_history[0].success
    assert isinstance(ctx_plugin.ctx, Ctx)
    assert kwargs_plugin.kwargs == {}


@pytest.mark.asyncio
async def test_step_updates_context_automatic_merge() -> None:
    @step(updates_context=True)
    async def init(_: str) -> Ctx:
        return Ctx(num=42)

    @step
    async def read(ctx_obj: Ctx, *, context: Optional[Ctx] = None) -> int:
        assert context is not None
        return context.num

    runner = create_test_flujo(init >> read, context_model=Ctx)
    result = await gather_result(runner, "input")

    assert result.final_pipeline_context.num == 42
    assert result.step_history[-1].output == 42
