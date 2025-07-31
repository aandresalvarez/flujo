import asyncio
import pytest
from unittest.mock import AsyncMock

from flujo.application.core.ultra_executor import (
    OptimizedExecutorCore,
    ObjectPool,
    OptimizedContextManager,
)
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent
from flujo.domain.models import UsageLimits
from flujo.exceptions import UsageLimitExceededError


def create_step(output: str = "ok") -> Step:
    return Step.model_validate({
        "name": "int_step",
        "agent": StubAgent([output]),
        "config": StepConfig(max_retries=1),
    })


@pytest.mark.asyncio
async def test_component_interface_optimization():
    core = OptimizedExecutorCore()
    assert isinstance(core._object_pool, ObjectPool)
    assert isinstance(core._context_manager_opt, OptimizedContextManager)


@pytest.mark.asyncio
async def test_dependency_injection_performance():
    class CustomRunner:
        async def run(self, *args, **kwargs):
            return "custom"
    core = OptimizedExecutorCore(agent_runner=CustomRunner())
    step = create_step()
    result = await core.optimized_execute(step, "data")
    assert result.output == "custom"


@pytest.mark.asyncio
async def test_component_lifecycle_optimization():
    pool = ObjectPool()
    obj = await pool.get(dict)
    await pool.put(obj)
    obj2 = await pool.get(dict)
    assert obj is obj2


@pytest.mark.asyncio
async def test_error_handling_optimization():
    async def bad_run(*_, **__):
        raise RuntimeError("fail")
    step = create_step()
    step.agent.run = AsyncMock(side_effect=bad_run)
    core = OptimizedExecutorCore()
    result = await core.optimized_execute(step, "data")
    assert not result.success


@pytest.mark.asyncio
async def test_concurrent_step_execution():
    core = OptimizedExecutorCore()
    step = create_step()
    results = await asyncio.gather(*(core.optimized_execute(step, i) for i in range(5)))
    assert len(results) == 5


@pytest.mark.asyncio
async def test_resource_management_optimization():
    core = OptimizedExecutorCore()
    step = create_step()
    result = await core.optimized_execute(step, "data")
    assert result.success


@pytest.mark.asyncio
async def test_usage_limit_enforcement_performance():
    core = OptimizedExecutorCore()
    await core._usage_meter.add(1.0, 0, 0)
    with pytest.raises(UsageLimitExceededError):
        await core._usage_meter.guard(UsageLimits(total_cost_usd_limit=0.5))


@pytest.mark.asyncio
async def test_telemetry_performance():
    core = OptimizedExecutorCore()

    @core._telemetry_opt.optimized_trace("demo")
    async def work():
        return "done"

    result = await work()
    assert result == "done"

