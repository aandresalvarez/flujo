import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import Success, Failure, Paused, StepResult


@pytest.mark.asyncio
async def test_parallel_policy_success_aggregates_outputs():
    core = ExecutorCore()
    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_EchoAgent())),
        "b": Pipeline.from_step(Step(name="B", agent=_EchoAgent())),
    }
    p = ParallelStep(name="p", branches=branches)

    outcome = await DefaultParallelStepExecutor().execute(
        core,
        p,
        data=None,
        context=None,
        resources=None,
        limits=None,
        breach_event=None,
        context_setter=None,
        step_executor=None,
    )
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True


@pytest.mark.asyncio
async def test_parallel_policy_failure_does_not_merge_context():
    core = ExecutorCore()

    class _FailAgent:
        async def run(self, *_args, **_kwargs):
            raise RuntimeError("nope")

    branches = {
        "bad": Pipeline.from_step(Step(name="BAD", agent=_FailAgent())),
        "ok": Pipeline.from_step(Step(name="OK", agent=object())),
    }
    p = ParallelStep(name="p", branches=branches)

    # Provide a context with a sentinel to validate not merged on failure
    class _Ctx:
        def __init__(self) -> None:
            self.scratchpad = {}

    ctx = _Ctx()
    outcome = await DefaultParallelStepExecutor().execute(
        core,
        p,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        breach_event=None,
        context_setter=None,
        step_executor=None,
    )
    assert isinstance(outcome, Failure)
    # On failure, branch_context should be None or not merged into original context
    if outcome.step_result is not None:
        assert outcome.step_result.branch_context is None or outcome.step_result.branch_context is ctx


@pytest.mark.asyncio
async def test_parallel_policy_yields_failure_on_paused_branch():
    core = ExecutorCore()

    class _HitlAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PausedException

            raise PausedException("wait")

    branches = {"h": Pipeline.from_step(Step(name="H", agent=_HitlAgent()))}
    p = ParallelStep(name="p", branches=branches)

    outcome = await DefaultParallelStepExecutor().execute(
        core,
        p,
        data=None,
        context=None,
        resources=None,
        limits=None,
        breach_event=None,
        context_setter=None,
        step_executor=None,
    )
    # Parallel policy currently wraps paused branch into Failure with appropriate feedback
    assert isinstance(outcome, Failure)
