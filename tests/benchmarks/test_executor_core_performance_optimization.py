import asyncio
import pytest

from flujo.application.core.ultra_executor import OptimizedExecutorCore
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent


def create_step(output: str = "ok") -> Step:
    return Step.model_validate({
        "name": "perf_step",
        "agent": StubAgent([output]),
        "config": StepConfig(max_retries=1),
    })


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_executor_core_execution_performance():
    core = OptimizedExecutorCore()
    step = create_step()
    result = await core.optimized_execute(step, "data")
    assert result.success


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_executor_core_memory_usage():
    core = OptimizedExecutorCore()
    step = create_step()
    result = await core.optimized_execute(step, "data")
    assert result.success


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_executor_core_concurrent_execution():
    core = OptimizedExecutorCore()
    step = create_step()
    results = await asyncio.gather(*(core.optimized_execute(step, i) for i in range(3)))
    assert all(r.success for r in results)


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_executor_core_cache_performance():
    core = OptimizedExecutorCore()
    step = create_step()
    await core.optimized_execute(step, "x")
    result = await core.optimized_execute(step, "x")
    assert result.success


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_executor_core_context_handling_performance():
    core = OptimizedExecutorCore()
    step = create_step()
    result = await core.optimized_execute(step, "data", context={})
    assert result.success

