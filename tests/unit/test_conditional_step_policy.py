import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultConditionalStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import Success, Failure, Paused, StepResult


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

    out = await DefaultConditionalStepExecutor().execute(
        core,
        cond,
        data={"branch": "h"},
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        _fallback_depth=0,
    )
    # Conditional policy translates Paused into Failure at this layer
    assert isinstance(out, Failure)
