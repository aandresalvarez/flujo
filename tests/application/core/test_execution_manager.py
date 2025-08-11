import asyncio
import pytest

from flujo.application.core.execution_manager import ExecutionManager
from flujo.domain.models import PipelineResult, StepResult, Success, Failure, Paused, Chunk
from flujo.domain.dsl.step import Step


class _FakePipeline:
    def __init__(self, steps):
        self.steps = steps


class _FakeStepCoordinator:
    def __init__(self, sequence):
        self.sequence = sequence

    async def execute_step(self, **_kwargs):
        for item in self.sequence:
            yield item

    def update_pipeline_result(self, result, step_result):
        result.step_history.append(step_result)
        result.total_cost_usd += step_result.cost_usd or 0.0
        result.total_tokens += step_result.token_counts or 0


@pytest.mark.asyncio
async def test_execution_manager_consumes_success_outcome():
    step = Step(name="s1", agent=object())
    pipeline = _FakePipeline([step])
    sr = StepResult(name="s1", success=True, output="ok")
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Success(step_result=sr)]))
    result = PipelineResult()
    # drain generator
    async for _ in em.execute_steps(0, data=None, context=None, result=result):
        pass
    assert len(result.step_history) == 1
    assert result.step_history[0].success is True


@pytest.mark.asyncio
async def test_execution_manager_stops_on_failure_outcome():
    step = Step(name="s1", agent=object())
    pipeline = _FakePipeline([step])
    sr = StepResult(name="s1", success=False, output=None, feedback="bad")
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Failure(error="x", feedback="bad", step_result=sr)]))
    result = PipelineResult()
    outs = []
    async for item in em.execute_steps(0, data=None, context=None, result=result):
        outs.append(item)
    # Failure should yield final result and stop
    assert len(result.step_history) == 1
    assert result.step_history[0].success is False


@pytest.mark.asyncio
async def test_execution_manager_raises_abort_on_paused_outcome():
    step = Step(name="s1", agent=object())
    pipeline = _FakePipeline([step])
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Paused(message="wait")]))
    result = PipelineResult()
    with pytest.raises(Exception):
        async for _ in em.execute_steps(0, data=None, context=None, result=result):
            pass

