import pytest
from typing import Any

from flujo.application.core.factories import ExecutorFactory
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.execution_dispatcher import ExecutionDispatcher
from flujo.application.core.policy_registry import PolicyRegistry, StepPolicy
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success, StepResult


class DummyStep(Step[str, str]):
    pass


class ChildStep(DummyStep):
    pass


class _FactoryPolicy(StepPolicy[DummyStep]):
    @property
    def handles_type(self) -> type[DummyStep]:
        return DummyStep

    async def execute(self, core: Any, frame: ExecutionFrame[object]) -> Success[StepResult]:
        return Success(step_result=StepResult(name=frame.step.name, success=True, output="factory"))


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


def test_policy_lookup_cache_invalidation_on_register() -> None:
    registry = PolicyRegistry()

    async def fallback(frame: ExecutionFrame[object]) -> Success[StepResult]:
        return Success(
            step_result=StepResult(name=frame.step.name, success=True, output="fallback")
        )

    async def parent_policy(frame: ExecutionFrame[object]) -> Success[StepResult]:
        return Success(step_result=StepResult(name=frame.step.name, success=True, output="parent"))

    registry.register_fallback(fallback)

    assert registry.get(ChildStep) is fallback

    registry.register(DummyStep, parent_policy)

    assert registry.get(ChildStep) is parent_policy


@pytest.mark.asyncio
async def test_executor_factory_policy_override_applied() -> None:
    factory = ExecutorFactory(policy_overrides=[_FactoryPolicy()])
    executor = factory.create_executor()

    step = DummyStep(name="factory-override")
    outcome = await executor.execute(step, data=None)

    result = outcome.step_result if isinstance(outcome, Success) else outcome

    assert isinstance(result, StepResult)
    assert result.output == "factory"
