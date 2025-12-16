import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultParallelStepExecutor
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.domain.dsl.step import Step
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import Success


@pytest.mark.asyncio
async def test_parallel_quota_split_with_infinite_cost_propagates_inf_to_children():
    core = ExecutorCore()

    class _EchoAgent:
        async def run(self, payload, context=None, resources=None, **kwargs):
            return payload

    branches = {
        "a": Pipeline.from_step(Step(name="A", agent=_EchoAgent())),
        "b": Pipeline.from_step(Step(name="B", agent=_EchoAgent())),
    }
    p = ParallelStep(name="p_inf", branches=branches)

    from flujo.domain.models import Quota

    parent_quota = Quota(remaining_cost_usd=float("inf"), remaining_tokens=10)
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
    assert rem_tokens == 10
    assert rem_cost == float("inf")
