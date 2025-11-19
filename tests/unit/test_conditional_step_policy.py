import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultConditionalStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import Success, Failure
from flujo.domain.dsl.parallel import ParallelStep


@pytest.mark.asyncio
async def test_conditional_policy_success_and_failure_paths():
    core = ExecutorCore()

    class _FailAgent:
        async def run(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    def pick_branch(data, ctx):
        return data.get("branch")

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    cond = ConditionalStep(
        name="c",
        condition_callable=pick_branch,
        branches={
            "ok": Pipeline.from_step(Step(name="OK", agent=_EchoAgent())),
            "bad": Pipeline.from_step(Step(name="BAD", agent=_FailAgent())),
        },
        default_branch_pipeline=Pipeline.from_step(Step(name="DEF", agent=_EchoAgent())),
    )

    # When data selects ok branch
    out1 = await DefaultConditionalStepExecutor().execute(
        core,
        cond,
        data={"branch": "ok"},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        _fallback_depth=0,
    )
    assert isinstance(out1, Success)

    # When data selects failing branch
    out2 = await DefaultConditionalStepExecutor().execute(
        core,
        cond,
        data={"branch": "bad"},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        _fallback_depth=0,
    )
    assert isinstance(out2, Failure)


@pytest.mark.asyncio
async def test_conditional_policy_returns_failure_on_paused_branch():
    core = ExecutorCore()

    class _HitlAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PausedException

            raise PausedException("wait")

    def pick_branch(data, ctx):
        return data.get("branch")

    cond = ConditionalStep(
        name="c",
        condition_callable=pick_branch,
        branches={"h": Pipeline.from_step(Step(name="H", agent=_HitlAgent()))},
        default_branch_pipeline=Pipeline.from_step(Step(name="H", agent=_HitlAgent())),
    )

    from flujo.exceptions import PausedException

    with pytest.raises(PausedException):
        await DefaultConditionalStepExecutor().execute(
            core,
            cond,
            data={"branch": "h"},
            context=None,
            resources=None,
            limits=None,
            context_setter=None,
            _fallback_depth=0,
        )


@pytest.mark.asyncio
async def test_conditional_branch_executes_parallel_with_quota_split():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    # Build a parallel step used inside a conditional branch
    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_EchoAgent())),
        "b": Pipeline.from_step(Step(name="B", agent=_EchoAgent())),
    }
    parallel = ParallelStep(name="p_in_cond", branches=branches)

    def pick_branch(data, ctx):
        return data.get("branch", "p")

    cond = ConditionalStep(
        name="cond",
        condition_callable=pick_branch,
        branches={
            "p": parallel,
        },
        default_branch_pipeline=Pipeline.from_step(Step(name="DEF", agent=_EchoAgent())),
    )

    out = await DefaultConditionalStepExecutor().execute(
        core,
        cond,
        data={"branch": "p"},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        _fallback_depth=0,
    )
    assert isinstance(out, Success)
