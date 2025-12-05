import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import (
    DefaultConditionalStepExecutor,
    DefaultDynamicRouterStepExecutor,
)
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import Success


@pytest.mark.asyncio
async def test_quota_propagates_conditional_to_nested_parallel_with_costs():
    core = ExecutorCore()

    class _CostAgent:
        def __init__(self, cost):
            self.cost = cost

        async def run(self, *_args, **_kwargs):
            class _Out:
                pass

            return _Out()

    # Nested parallel inside a conditional branch
    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_CostAgent(1.0))),
        "b": Pipeline.from_step(Step(name="B", agent=_CostAgent(2.0))),
    }
    inner_parallel = ParallelStep(name="p_inner", branches=branches)

    def pick_branch(data, ctx):
        return "inner"

    cond = ConditionalStep(
        name="cond",
        condition_callable=pick_branch,
        branches={"inner": inner_parallel},
        default_branch_pipeline=None,
    )

    # Patch usage metrics globally for this test
    import flujo.cost as cost_mod
    import flujo.application.core.step_policies as policies_mod

    real_extract = cost_mod.extract_usage_metrics
    real_extract2 = policies_mod.extract_usage_metrics

    def fake_extract_usage_metrics(raw_output, agent, step_name):
        if step_name == "A":
            return (0, 0, 1.0)
        if step_name == "B":
            return (0, 0, 2.0)
        return (0, 0, 0.0)

    cost_mod.extract_usage_metrics = fake_extract_usage_metrics
    policies_mod.extract_usage_metrics = fake_extract_usage_metrics
    try:
        # Provide sufficient quota on the core
        from flujo.domain.models import Quota

        core._set_current_quota(Quota(remaining_cost_usd=5.0, remaining_tokens=1000))

        frame = make_execution_frame(
            core,
            cond,
            {},
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
        outcome = await DefaultConditionalStepExecutor().execute(core=core, frame=frame)
        assert isinstance(outcome, Success)
        # Validate aggregated cost from inner parallel branches
        assert abs(outcome.step_result.cost_usd - 3.0) < 1e-6
    finally:
        cost_mod.extract_usage_metrics = real_extract
        policies_mod.extract_usage_metrics = real_extract2


@pytest.mark.asyncio
async def test_quota_propagates_router_to_selected_parallel_with_costs():
    core = ExecutorCore()

    class _CostAgent:
        def __init__(self, cost):
            self.cost = cost

        async def run(self, *_args, **_kwargs):
            class _Out:
                pass

            return _Out()

    class _RouterAgent:
        async def run(self, *_args, **_kwargs):
            # Select two branches
            return ["x", "y"]

    branches = {
        "x": Pipeline.from_step(Step(name="X", agent=_CostAgent(1.0))),
        "y": Pipeline.from_step(Step(name="Y", agent=_CostAgent(3.0))),
        "z": Pipeline.from_step(Step(name="Z", agent=_CostAgent(0.0))),
    }

    from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep

    router = DynamicParallelRouterStep(
        name="router",
        router_agent=_RouterAgent(),
        branches=branches,
    )

    # Patch usage metrics
    import flujo.cost as cost_mod
    import flujo.application.core.step_policies as policies_mod

    real_extract = cost_mod.extract_usage_metrics
    real_extract2 = policies_mod.extract_usage_metrics

    def fake_extract_usage_metrics(raw_output, agent, step_name):
        if step_name == "X":
            return (0, 0, 1.0)
        if step_name == "Y":
            return (0, 0, 3.0)
        if step_name == "Z":
            return (0, 0, 0.0)
        return (0, 0, 0.0)

    cost_mod.extract_usage_metrics = fake_extract_usage_metrics
    policies_mod.extract_usage_metrics = fake_extract_usage_metrics
    try:
        from flujo.domain.models import Quota

        core._set_current_quota(Quota(remaining_cost_usd=10.0, remaining_tokens=1000))

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
        assert outcome.step_result.metadata_["executed_branches"] == ["x", "y"]
    finally:
        cost_mod.extract_usage_metrics = real_extract
        policies_mod.extract_usage_metrics = real_extract2
