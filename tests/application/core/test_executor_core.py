from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step
from flujo.domain.models import Failure, StepResult


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
        breach_event,
        _fallback_depth=0,
    ):
        raise ValueError("boom")


async def test_executor_core_choke_point_converts_unexpected_exception_to_failure():
    core = ExecutorCore()
    # Inject a policy that raises unexpectedly
    core.agent_step_executor = _RaisingAgentStepExecutor()

    step = Step(name="unit", agent=object())

    outcome = await core.execute(step=step, data="x")
    assert isinstance(outcome, Failure)
    assert isinstance(outcome.error, Exception)
    assert "boom" in (outcome.feedback or "")
    assert isinstance(outcome.step_result, StepResult)
    assert outcome.step_result.success is False
