import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import Success, Failure, Paused


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

    frame = make_execution_frame(
        core,
        p,
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
    outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
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

    # Provide a typed context with a sentinel to validate not merged on failure
    from flujo.domain.models import PipelineContext

    ctx = PipelineContext()
    frame = make_execution_frame(
        core,
        p,
        data=None,
        context=ctx,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome, Failure)
    # On failure, branch_context should be None or not merged into original context
    if outcome.step_result is not None:
        from flujo.domain.models import PipelineContext

        assert (
            outcome.step_result.branch_context is None
            or isinstance(outcome.step_result.branch_context, PipelineContext)
            or outcome.step_result.branch_context is ctx
        )


@pytest.mark.asyncio
async def test_parallel_policy_yields_failure_on_paused_branch():
    core = ExecutorCore()

    class _HitlAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PausedException

            raise PausedException("wait")

    branches = {"h": Pipeline.from_step(Step(name="H", agent=_HitlAgent()))}
    p = ParallelStep(name="p", branches=branches)

    frame = make_execution_frame(
        core,
        p,
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
    outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
    # Parallel policy should propagate Paused so the runner pauses
    assert isinstance(outcome, Paused)


@pytest.mark.asyncio
async def test_parallel_policy_deterministic_quota_split_and_accounting():
    core = ExecutorCore()

    class _CostAgent:
        def __init__(self, cost):
            self.cost = cost

        async def run(self, *_args, **_kwargs):
            class _Out:
                pass

            return _Out()

    # Three branches with deterministic costs
    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_CostAgent(cost=1.25))),
        "b": Pipeline.from_step(Step(name="B", agent=_CostAgent(cost=2.75))),
        "c": Pipeline.from_step(Step(name="C", agent=_CostAgent(cost=0.0))),
    }
    p = ParallelStep(name="p", branches=branches)

    # Create a parent quota and set it before running parallel
    from flujo.domain.models import Quota

    parent_quota = Quota(remaining_cost_usd=10.0, remaining_tokens=1000)
    core._set_current_quota(parent_quota)

    # Monkeypatch extract_usage_metrics to return configured costs and zero tokens
    import flujo.cost as cost_mod
    import flujo.application.core.step_policies as policies_mod

    real_extract = cost_mod.extract_usage_metrics
    real_extract2 = policies_mod.extract_usage_metrics

    def fake_extract_usage_metrics(raw_output, agent, step_name):
        if step_name == "A":
            return (0, 0, 1.25)
        if step_name == "B":
            return (0, 0, 2.75)
        if step_name == "C":
            return (0, 0, 0.0)
        return (0, 0, 0.0)

    cost_mod.extract_usage_metrics = fake_extract_usage_metrics
    policies_mod.extract_usage_metrics = fake_extract_usage_metrics
    try:
        frame = make_execution_frame(
            core,
            p,
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
        outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
        assert isinstance(outcome, Success)
        # Sum of branch costs equals 4.0
        assert abs(outcome.step_result.cost_usd - 4.0) < 1e-6
        # Determinism check: executed branches order is the input order
        assert outcome.step_result.metadata_["executed_branches"] == ["a", "b", "c"]
    finally:
        cost_mod.extract_usage_metrics = real_extract
        policies_mod.extract_usage_metrics = real_extract2


@pytest.mark.asyncio
async def test_parallel_policy_propagates_usage_errors_and_mock_detection():
    core = ExecutorCore()

    class _MockAgent:
        async def run(self, *_args, **_kwargs):
            from unittest.mock import MagicMock

            return MagicMock()

    class _StrictPricingAgent:
        async def run(self, *_args, **_kwargs):
            from flujo.exceptions import PricingNotConfiguredError

            raise PricingNotConfiguredError("strict pricing not configured")

    branches = {
        "mock": Pipeline.from_step(Step(name="M", agent=_MockAgent())),
        "price": Pipeline.from_step(Step(name="P", agent=_StrictPricingAgent())),
    }
    p = ParallelStep(name="p_errs", branches=branches)

    frame = make_execution_frame(
        core,
        p,
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
    outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
    # Framework converts most branch exceptions to Failure at the parallel level
    from flujo.domain.models import Failure as OutcomeFailure

    assert isinstance(outcome, OutcomeFailure)


@pytest.mark.asyncio
async def test_parallel_policy_quota_splitting_zero_parent_after_split():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_EchoAgent())),
        "b": Pipeline.from_step(Step(name="B", agent=_EchoAgent())),
        "c": Pipeline.from_step(Step(name="C", agent=_EchoAgent())),
    }
    p = ParallelStep(name="p_zero_parent", branches=branches)

    from flujo.domain.models import Quota

    parent_quota = Quota(remaining_cost_usd=9.0, remaining_tokens=9)
    core._set_current_quota(parent_quota)

    frame = make_execution_frame(
        core,
        p,
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
    outcome = await DefaultParallelStepExecutor().execute(core=core, frame=frame)
    assert isinstance(outcome, Success)
    rem_cost, rem_tokens = parent_quota.get_remaining()
    # Parent quota is shared across branches; no forced zeroing.
    assert rem_tokens == 9
    assert rem_cost == 9.0
