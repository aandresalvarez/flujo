import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.models import StepResult, StepOutcome, Success
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step


class _FakeParallelExecutor:
    async def execute(self, core, frame) -> StepResult:
        step = frame.step
        return StepResult(name=getattr(step, "name", "parallel"), success=True, output={"ok": True})


class _FakeConditionalExecutor:
    async def execute(self, core, frame) -> StepResult:
        step = frame.step
        return StepResult(name=getattr(step, "name", "conditional"), success=True, output=42)


@pytest.mark.asyncio
async def test_parallel_adapter_returns_outcome_in_backend_path():
    core = ExecutorCore()
    core.parallel_step_executor = _FakeParallelExecutor()
    step = ParallelStep(name="p_test", branches={"noop": Pipeline.from_step(Step(name="noop"))})
    frame = ExecutionFrame(
        step=step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )

    outcome = await core.execute(frame)

    assert isinstance(outcome, StepOutcome)
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True
    assert outcome.step_result.output == {"ok": True}


@pytest.mark.asyncio
async def test_conditional_adapter_returns_outcome_in_backend_path():
    core = ExecutorCore()
    core.conditional_step_executor = _FakeConditionalExecutor()

    def _always_a(_data, _ctx):
        return "a"

    from flujo.domain.dsl.pipeline import Pipeline

    step = ConditionalStep(
        name="c_test", condition_callable=_always_a, branches={"a": Pipeline(steps=[])}
    )
    frame = ExecutionFrame(
        step=step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )

    outcome = await core.execute(frame)

    assert isinstance(outcome, StepOutcome)
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True
    assert outcome.step_result.output == 42
