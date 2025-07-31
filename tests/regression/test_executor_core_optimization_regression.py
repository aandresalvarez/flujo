import pytest
from unittest.mock import AsyncMock

from flujo.application.core.ultra_executor import ExecutorCore, OptimizedExecutorCore
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent


def create_step(output: str = "ok") -> Step:
    return Step.model_validate({
        "name": "reg_step",
        "agent": StubAgent([output]),
        "config": StepConfig(max_retries=1),
    })


@pytest.mark.asyncio
async def test_optimization_functionality_preservation():
    base = ExecutorCore()
    opt = OptimizedExecutorCore()
    step_base = create_step()
    step_opt = create_step()
    result_base = await base.execute(step_base, "data")
    result_opt = await opt.optimized_execute(step_opt, "data")
    assert result_base.success == result_opt.success
    assert result_base.output == result_opt.output


@pytest.mark.asyncio
async def test_optimization_backward_compatibility():
    opt = OptimizedExecutorCore()
    step = create_step()
    result = await opt.execute(step, "data")
    assert result.success


@pytest.mark.asyncio
async def test_optimization_error_handling():
    step = create_step()
    step.agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    opt = OptimizedExecutorCore()
    result = await opt.optimized_execute(step, "data")
    assert not result.success

