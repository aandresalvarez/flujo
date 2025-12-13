import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultDynamicRouterStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import Success


@pytest.mark.asyncio
async def test_dynamic_router_paused_returns_paused():
    core = ExecutorCore()

    class _PausedRouterAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PausedException

            raise PausedException("wait")

    # Router that pauses
    from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

    router = DynamicParallelRouterStep(
        name="router",
        router_agent=_PausedRouterAgent(),
        branches={"noop": Pipeline.from_step(Step(name="noop", agent=object()))},
    )

    frame = make_execution_frame(
        core,
        router,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    from flujo.exceptions import PausedException

    with pytest.raises(PausedException):
        await DefaultDynamicRouterStepExecutor().execute(core, frame)


@pytest.mark.asyncio
async def test_dynamic_router_executes_selected_branches_in_order():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    class _RouterAgent:
        async def run(self, *_args, **_kwargs):
            return ["a", "b"]

    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_EchoAgent())),
        "b": Pipeline.from_step(Step(name="B", agent=_EchoAgent())),
        "c": Pipeline.from_step(Step(name="C", agent=_EchoAgent())),
    }

    from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

    router = DynamicParallelRouterStep(
        name="router",
        router_agent=_RouterAgent(),
        branches=branches,
    )

    frame = make_execution_frame(
        core,
        router,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultDynamicRouterStepExecutor().execute(core, frame)
    assert isinstance(outcome, Success)
    assert outcome.step_result.metadata_["executed_branches"] == ["a", "b"]


@pytest.mark.asyncio
async def test_dynamic_router_no_selected_branches_returns_empty():
    core = ExecutorCore()

    class _RouterAgent:
        async def run(self, *_args, **_kwargs):
            return []

    from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

    router = DynamicParallelRouterStep(
        name="router",
        router_agent=_RouterAgent(),
        branches={
            "x": Pipeline.from_step(Step(name="X", agent=object())),
        },
    )

    frame = make_execution_frame(
        core,
        router,
        data=None,
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultDynamicRouterStepExecutor().execute(core, frame)
    assert isinstance(outcome, Success)
    assert isinstance(outcome.step_result.output, dict)
    assert outcome.step_result.output == {}
