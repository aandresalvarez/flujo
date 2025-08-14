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
    em = ExecutionManager(
        pipeline, step_coordinator=_FakeStepCoordinator([Success(step_result=sr)])
    )
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
    em = ExecutionManager(
        pipeline,
        step_coordinator=_FakeStepCoordinator([Failure(error="x", feedback="bad", step_result=sr)]),
    )
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


@pytest.mark.asyncio
async def test_execution_manager_passes_through_chunk_and_aborted():
    step = Step(name="s1", agent=object())
    pipeline = _FakePipeline([step])
    from flujo.domain.models import Aborted

    seq = [Chunk(data={"x": 1}), Aborted(reason="stop"), StepResult(name="s1", success=True)]
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator(seq))
    result = PipelineResult()
    items = []
    async for it in em.execute_steps(0, data=None, context=None, result=result):
        items.append(it)
    assert isinstance(items[0], Chunk)
    assert isinstance(items[-1], PipelineResult)

@pytest.mark.asyncio
async def test_execution_manager_accumulates_costs_and_tokens_over_multiple_successes():
    # Given multiple steps that each produce a successful StepResult with cost and token counts
    s1 = Step(name="s1", agent=object())
    s2 = Step(name="s2", agent=object())
    pipeline = _FakePipeline([s1, s2])

    sr1 = StepResult(name="s1", success=True, output="o1", cost_usd=0.12, token_counts=120)
    sr2 = StepResult(name="s2", success=True, output="o2", cost_usd=1.23, token_counts=880)

    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Success(step_result=sr1), Success(step_result=sr2)]))
    result = PipelineResult()

    # When
    drained = []
    async for it in em.execute_steps(0, data={"key": "value"}, context={"user": "u"}, result=result):
        drained.append(it)

    # Then: we expect a final PipelineResult yielded with total aggregation
    assert isinstance(drained[-1], PipelineResult)
    assert len(result.step_history) == 2
    assert result.total_cost_usd == pytest.approx(1.35)
    assert result.total_tokens == 1000
    assert result.step_history[0].output == "o1"
    assert result.step_history[1].output == "o2"


@pytest.mark.asyncio
async def test_execution_manager_handles_none_costs_and_token_counts_gracefully():
    # Given a step result with None costs and tokens, aggregation should not break and not increment totals
    s1 = Step(name="s1", agent=object())
    pipeline = _FakePipeline([s1])
    sr = StepResult(name="s1", success=True, output="ok", cost_usd=None, token_counts=None)
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Success(step_result=sr)]))
    result = PipelineResult()

    async for _ in em.execute_steps(0, data=None, context=None, result=result):
        pass

    assert len(result.step_history) == 1
    assert result.total_cost_usd == pytest.approx(0.0)
    assert result.total_tokens == 0


@pytest.mark.asyncio
async def test_execution_manager_stops_on_failure_and_final_result_contains_failure_feedback():
    # Given a failure after a success, ensure execution stops and final result reflects failure
    s1 = Step(name="s1", agent=object())
    s2 = Step(name="s2", agent=object())
    pipeline = _FakePipeline([s1, s2])
    ok_sr = StepResult(name="s1", success=True, output="ok")
    bad_sr = StepResult(name="s2", success=False, output=None, feedback="something bad")

    em = ExecutionManager(
        pipeline,
        step_coordinator=_FakeStepCoordinator([Success(step_result=ok_sr), Failure(error="boom", feedback="something bad", step_result=bad_sr)])
    )
    result = PipelineResult()
    items = []
    async for it in em.execute_steps(0, data=None, context=None, result=result):
        items.append(it)

    # Failure should stop additional steps and yield final result
    assert isinstance(items[-1], PipelineResult)
    assert len(result.step_history) == 2
    assert result.step_history[0].success is True
    assert result.step_history[1].success is False
    assert result.step_history[1].feedback == "something bad"


@pytest.mark.asyncio
async def test_execution_manager_direct_step_result_emission_is_handled():
    # Some step coordinators may yield StepResult directly (not wrapped in Success/Failure)
    s1 = Step(name="s1", agent=object())
    pipeline = _FakePipeline([s1])
    direct_sr = StepResult(name="s1", success=True, output="raw-ok", cost_usd=0.5, token_counts=55)

    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([direct_sr]))
    result = PipelineResult()
    outs = []
    async for out in em.execute_steps(0, data=None, context=None, result=result):
        outs.append(out)

    # Should still produce a PipelineResult and record the step
    assert isinstance(outs[-1], PipelineResult)
    assert len(result.step_history) == 1
    assert result.step_history[0].output == "raw-ok"
    assert result.total_cost_usd == pytest.approx(0.5)
    assert result.total_tokens == 55


@pytest.mark.asyncio
async def test_execution_manager_emits_chunks_interleaved_with_successes_and_yields_final_result():
    # Given streaming chunks between step outcomes
    s1 = Step(name="s1", agent=object())
    pipeline = _FakePipeline([s1])
    sr1 = StepResult(name="s1", success=True, output="done")

    seq = [Chunk(data={"partial": 1}), Chunk(data="...), user typing"), Success(step_result=sr1)]
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator(seq))
    result = PipelineResult()

    streamed = []
    async for item in em.execute_steps(0, data=None, context=None, result=result):
        streamed.append(item)

    # Expect first items to be chunks and the last one to be final PipelineResult
    assert isinstance(streamed[0], Chunk)
    assert isinstance(streamed[1], Chunk)
    assert isinstance(streamed[-1], PipelineResult)
    assert len(result.step_history) == 1
    assert result.step_history[0].success is True


@pytest.mark.asyncio
async def test_execution_manager_aborted_early_yields_final_result_without_processing_further():
    from flujo.domain.models import Aborted

    s1 = Step(name="s1", agent=object())
    s2 = Step(name="s2", agent=object())
    pipeline = _FakePipeline([s1, s2])
    sr1 = StepResult(name="s1", success=True, output="ok1")
    # Abort before second step success
    seq = [Success(step_result=sr1), Aborted(reason="user_stop")]
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator(seq))
    result = PipelineResult()

    collected = []
    async for it in em.execute_steps(0, data=None, context=None, result=result):
        collected.append(it)

    # After Aborted, should produce final result and stop
    assert isinstance(collected[-1], PipelineResult)
    assert len(result.step_history) == 1
    assert result.step_history[0].output == "ok1"


@pytest.mark.asyncio
async def test_execution_manager_paused_raises_abort_or_specific_exception_type():
    # The implementation under test should raise upon Paused. We assert it raises an exception.
    s1 = Step(name="s1", agent=object())
    pipeline = _FakePipeline([s1])
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Paused(message="hold please")]))

    result = PipelineResult()
    with pytest.raises(Exception):
        async for _ in em.execute_steps(0, data=None, context=None, result=result):
            pass


@pytest.mark.asyncio
async def test_execution_manager_unsupported_outcome_raises_exception():
    # Create a dummy unsupported type to verify robustness
    class _Weird:
        def __init__(self, x):
            self.x = x

    s1 = Step(name="s1", agent=object())
    pipeline = _FakePipeline([s1])
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([_Weird(x=42)]))
    result = PipelineResult()

    # We expect the manager to raise on unknown outcome types
    with pytest.raises(Exception):
        async for _ in em.execute_steps(0, data=None, context=None, result=result):
            pass


@pytest.mark.asyncio
async def test_execution_manager_yields_only_one_final_pipeline_result_per_run():
    # Ensure only one PipelineResult is yielded, even for multiple steps success
    s1 = Step(name="s1", agent=object())
    s2 = Step(name="s2", agent=object())
    pipeline = _FakePipeline([s1, s2])

    sr1 = StepResult(name="s1", success=True, output="ok1")
    sr2 = StepResult(name="s2", success=True, output="ok2")
    em = ExecutionManager(pipeline, step_coordinator=_FakeStepCoordinator([Success(step_result=sr1), Success(step_result=sr2)]))
    result = PipelineResult()

    yielded = []
    async for it in em.execute_steps(0, data=None, context=None, result=result):
        yielded.append(it)

    # Verify only one PipelineResult at end
    pipeline_results = [x for x in yielded if isinstance(x, PipelineResult)]
    assert len(pipeline_results) == 1
    assert len(result.step_history) == 2
    assert result.step_history[-1].output == "ok2"
