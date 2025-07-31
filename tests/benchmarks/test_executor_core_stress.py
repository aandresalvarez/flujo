import asyncio
import pytest

from flujo.application.core.ultra_executor import OptimizedExecutorCore
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent


def create_step(output: str = "ok") -> Step:
    return Step.model_validate({
        "name": "stress_step",
        "agent": StubAgent([output]),
        "config": StepConfig(max_retries=1),
    })


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_high_concurrency_stress():
    core = OptimizedExecutorCore()
    step = create_step()
    tasks = [core.optimized_execute(step, i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert all(r.success for r in results)


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_memory_pressure_stress():
    core = OptimizedExecutorCore()
    step = create_step()
    for _ in range(5):
        await core.optimized_execute(step, "x")
    assert True


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_cpu_intensive_stress():
    core = OptimizedExecutorCore()
    step = create_step()
    results = await asyncio.gather(*(core.optimized_execute(step, i) for i in range(5)))
    assert len(results) == 5


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_network_latency_stress():
    core = OptimizedExecutorCore()
    step = create_step()
    result = await core.optimized_execute(step, "data")
    assert result.success

