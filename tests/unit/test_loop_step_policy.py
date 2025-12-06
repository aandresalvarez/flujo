import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import (
    DefaultLoopStepExecutor,
    DefaultParallelStepExecutor,
)
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import Success, Failure


@pytest.mark.asyncio
async def test_loop_policy_returns_failure_outcome_on_iteration_mapper_error():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    body = Pipeline.from_step(Step(name="echo", agent=_EchoAgent()))

    # iteration_input_mapper will raise
    def bad_iter_mapper(output, ctx, i):
        raise RuntimeError("bad-map")

    loop = LoopStep(
        name="loop",
        loop_body_pipeline=body,
        # Force an iteration so the iteration_input_mapper is invoked
        exit_condition_callable=lambda _o, _c: False,
        iteration_input_mapper=bad_iter_mapper,
        max_loops=2,
    )

    frame = make_execution_frame(
        core,
        loop,
        data="in",
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
    outcome = await DefaultLoopStepExecutor().execute(core, frame)
    assert isinstance(outcome, Failure)
    assert "bad-map" in (outcome.feedback or "") or (
        outcome.step_result and "bad-map" in (outcome.step_result.feedback or "")
    )


@pytest.mark.asyncio
async def test_loop_policy_success_simple_exit():
    core = ExecutorCore()

    # Body returns immediately and exit condition true
    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    body = Pipeline.from_step(Step(name="unit", agent=_EchoAgent()))
    loop = LoopStep(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )
    frame = make_execution_frame(
        core,
        loop,
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
    outcome = await DefaultLoopStepExecutor().execute(core, frame)
    assert isinstance(outcome, Success)


@pytest.mark.asyncio
async def test_loop_within_parallel_and_parallel_within_loop_quota_composition():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    # Inner parallel inside loop
    inner_branches = {
        "x": Pipeline.from_step(Step(name="X", agent=_EchoAgent())),
        "y": Pipeline.from_step(Step(name="Y", agent=_EchoAgent())),
    }
    inner_parallel = ParallelStep(name="inner", branches=inner_branches)
    loop_body = Pipeline.from_step(Step(name="Pass", agent=_EchoAgent())) >> inner_parallel

    loop = LoopStep(
        name="outer_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )

    frame_loop = make_execution_frame(
        core,
        loop,
        data={"v": 1},
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
    outcome1 = await DefaultLoopStepExecutor().execute(core, frame_loop)
    assert isinstance(outcome1, Success)

    # Parallel containing loop in one branch
    loop2 = LoopStep(
        name="inner_loop",
        loop_body_pipeline=Pipeline.from_step(Step(name="LPass", agent=_EchoAgent())),
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )
    branches2 = {
        "l": Pipeline.from_step(Step(name="LP", agent=_EchoAgent())) >> loop2,
        "z": Pipeline.from_step(Step(name="Z", agent=_EchoAgent())),
    }
    p2 = ParallelStep(name="outer_p", branches=branches2)

    frame = make_execution_frame(
        core,
        p2,
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
    outcome2 = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome2, Success)


@pytest.mark.asyncio
async def test_loop_policy_raises_paused_exception():
    core = ExecutorCore()

    class _HitlAgent:
        async def run(self, *_, **__):
            from flujo.exceptions import PausedException

            raise PausedException("wait")

    body = Pipeline.from_step(Step(name="hitl", agent=_HitlAgent()))
    loop = LoopStep(
        name="loop",
        loop_body_pipeline=body,
        exit_condition_callable=lambda _o, _c: False,
        max_loops=1,
    )
    import pytest
    from flujo.exceptions import PausedException

    with pytest.raises(PausedException):
        frame = make_execution_frame(
            core,
            loop,
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
        _ = await DefaultLoopStepExecutor().execute(core, frame)
