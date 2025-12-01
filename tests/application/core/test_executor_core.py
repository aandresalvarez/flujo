from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.dynamic_router import DynamicParallelRouterStep
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult
from tests.test_types.fixtures import create_test_step


class _RaisingAgentStepExecutor:
    async def execute(
        self,
        core,
        step,
        data,
        context,
        resources,
        limits,
        stream,
        on_chunk,
        cache_key,
        _fallback_depth=0,
    ):
        raise ValueError("boom")


async def test_executor_core_choke_point_converts_unexpected_exception_to_failure():
    core = ExecutorCore()
    # Inject a policy that raises unexpectedly
    core.agent_step_executor = _RaisingAgentStepExecutor()

    step = create_test_step(name="unit", agent=object())

    outcome = await core.execute(step=step, data="x")
    assert isinstance(outcome, StepResult)
    assert outcome.success is False
    assert "boom" in (outcome.feedback or "")


def test_executor_core_policy_registry_populated():
    core = ExecutorCore()

    def _same_callable(a, b):
        return getattr(a, "__func__", a) is getattr(b, "__func__", b)

    # Base mapping
    assert _same_callable(core.policy_registry.get(Step), core._policy_default_step)
    # Complex steps
    assert _same_callable(core.policy_registry.get(LoopStep), core._policy_loop_step)
    assert _same_callable(core.policy_registry.get(ParallelStep), core._policy_parallel_step)
    assert _same_callable(core.policy_registry.get(ConditionalStep), core._policy_conditional_step)
    assert _same_callable(
        core.policy_registry.get(DynamicParallelRouterStep), core._policy_dynamic_router_step
    )
    assert _same_callable(core.policy_registry.get(HumanInTheLoopStep), core._policy_hitl_step)
