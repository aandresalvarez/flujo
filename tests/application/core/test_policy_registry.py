import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.execution_dispatcher import ExecutionDispatcher
from flujo.application.core.policy_registry import PolicyRegistry
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success, StepResult


class DummyStep(Step[str, str]):
    pass


@pytest.mark.asyncio
async def test_custom_policy_injection_with_executor_core() -> None:
    registry = PolicyRegistry()

    async def custom_policy(frame: ExecutionFrame[object]) -> Success[StepResult]:
        return Success(step_result=StepResult(name=frame.step.name, success=True, output="ok"))

    registry.register(DummyStep, custom_policy)
    core = ExecutorCore(policy_registry=registry)

    step = DummyStep(name="dummy")
    outcome = await core.execute(step, data="payload")

    if isinstance(outcome, Success):
        result = outcome.step_result
    else:
        result = outcome

    assert isinstance(result, StepResult)
    assert result.success
    assert result.output == "ok"


@pytest.mark.asyncio
async def test_fallback_policy_is_used_for_unregistered_step() -> None:
    registry = PolicyRegistry()

    async def fallback(frame: ExecutionFrame[object]) -> Success[StepResult]:
        return Success(
            step_result=StepResult(name=frame.step.name, success=True, output="fallback")
        )

    registry.register_fallback(fallback)
    dispatcher = ExecutionDispatcher(registry)

    step = DummyStep(name="unknown")
    frame = ExecutionFrame(
        step=step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _res, _ctx: None,
        quota=None,
        result=None,
        _fallback_depth=0,
    )

    outcome = await dispatcher.dispatch(frame)

    if isinstance(outcome, Success):
        result = outcome.step_result
    else:
        result = outcome

    assert isinstance(result, StepResult)
    assert result.output == "fallback"
