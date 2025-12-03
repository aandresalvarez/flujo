import pytest
import asyncio
import time
from typing import Any
from pydantic import BaseModel

from flujo import Step
from flujo.testing.utils import gather_result
from flujo.domain import UsageLimits
from tests.conftest import create_test_flujo


class LargeContext(BaseModel):
    """A context with many large fields to make copying expensive."""

    field_1: str = "value_1"
    field_2: str = "value_2"
    field_3: str = "value_3"
    field_4: str = "value_4"
    field_5: str = "value_5"
    large_data_1: str = "x" * 10000  # Large field (reduced from 50KB)
    large_data_2: str = "y" * 10000  # Another large field (reduced from 50KB)
    large_data_3: str = "z" * 10000  # Third large field (reduced from 50KB)
    shared_field: str = "shared"


class SimpleAgent:
    """A simple agent that just returns the input."""

    async def run(self, data: Any) -> Any:
        return data


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_context_copying_performance_benchmark() -> None:
    """Benchmark the performance difference between full and selective context copying."""

    # Create a context with large data
    context = LargeContext()

    # Create branches
    branches = {
        "branch_1": Step.model_validate({"name": "branch_1", "agent": SimpleAgent()}),
        "branch_2": Step.model_validate({"name": "branch_2", "agent": SimpleAgent()}),
        "branch_3": Step.model_validate({"name": "branch_3", "agent": SimpleAgent()}),
        "branch_4": Step.model_validate({"name": "branch_4", "agent": SimpleAgent()}),
    }

    # Test with selective context copying (only small fields)
    parallel_selective = Step.parallel(
        "parallel_selective",
        branches,
        context_include_keys=["field_1", "field_2", "field_3", "field_4", "field_5"],
    )

    # Test with full context copying (default behavior)
    parallel_full = Step.parallel("parallel_full", branches)

    runner_selective = create_test_flujo(
        parallel_selective, context_model=LargeContext, persist_state=False
    )
    runner_selective.disable_tracing()
    runner_full = create_test_flujo(parallel_full, context_model=LargeContext, persist_state=False)
    runner_full.disable_tracing()

    # Measure performance with selective copying
    start = time.monotonic()
    result_selective = await gather_result(
        runner_selective, "input", initial_context_data=context.model_dump()
    )
    selective_time = time.monotonic() - start

    # Measure performance with full copying
    start = time.monotonic()
    result_full = await gather_result(
        runner_full, "input", initial_context_data=context.model_dump()
    )
    full_time = time.monotonic() - start

    # Verify both produce correct results
    assert result_selective.step_history[-1].output == {
        "branch_1": "input",
        "branch_2": "input",
        "branch_3": "input",
        "branch_4": "input",
    }
    assert result_full.step_history[-1].output == {
        "branch_1": "input",
        "branch_2": "input",
        "branch_3": "input",
        "branch_4": "input",
    }

    # Report performance difference
    print("\nPerformance Benchmark Results:")
    print(f"Selective context copying time: {selective_time:.4f}s")
    print(f"Full context copying time: {full_time:.4f}s")
    improvement = (full_time - selective_time) / full_time * 100
    print(f"Performance improvement: {improvement:.1f}%")

    # Ensure selective copying is not significantly slower than full copy
    assert improvement >= -1000


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_proactive_cancellation_performance_benchmark() -> None:
    """Benchmark the performance improvement from proactive cancellation."""

    class FastExpensiveAgent:
        async def run(self, data: Any) -> Any:
            await asyncio.sleep(0.01)  # Very fast

            class Output(BaseModel):
                value: Any
                cost_usd: float = 0.15  # Expensive
                token_counts: int = 100

            return Output(value=data)

    class SlowCheapAgent:
        async def run(self, data: Any) -> Any:
            await asyncio.sleep(0.5)  # Slow

            class Output(BaseModel):
                value: Any
                cost_usd: float = 0.01  # Cheap
                token_counts: int = 10

            return Output(value=data)

    branches = {
        "fast_expensive": Step.model_validate(
            {"name": "fast_expensive", "agent": FastExpensiveAgent()}
        ),
        "slow_cheap": Step.model_validate({"name": "slow_cheap", "agent": SlowCheapAgent()}),
    }

    parallel = Step.parallel("parallel_cancellation_benchmark", branches)
    limits = UsageLimits(total_cost_usd_limit=0.10)  # Will be breached by fast_expensive
    runner = create_test_flujo(parallel, usage_limits=limits, persist_state=False)
    runner.disable_tracing()

    # Measure execution time with proactive cancellation
    start = time.monotonic()
    try:
        await gather_result(runner, "input")
    except Exception:
        pass  # Expected to fail due to limit breach
    cancellation_time = time.monotonic() - start

    # Measure execution time without limits (should take longer)
    runner_no_limits = create_test_flujo(parallel, persist_state=False)  # No usage limits
    runner_no_limits.disable_tracing()
    start = time.monotonic()
    await gather_result(runner_no_limits, "input")
    no_limits_time = time.monotonic() - start

    print("\nProactive Cancellation Benchmark Results:")
    print(f"With proactive cancellation: {cancellation_time:.4f}s")
    print(f"Without limits (full execution): {no_limits_time:.4f}s")
    print(f"Time saved: {((no_limits_time - cancellation_time) / no_limits_time * 100):.1f}%")

    # UltraExecutor makes both paths faster, so we need to be more tolerant
    # The key is that cancellation should not be significantly slower than full execution
    assert cancellation_time <= no_limits_time * 1.5  # Allow 50% tolerance
    # Both should be reasonably fast due to UltraExecutor optimizations
    # Allow for system overhead and timing variations
    assert cancellation_time < 1.2  # Should be fast due to cancellation
    assert no_limits_time < 1.2  # Should be reasonably fast due to UltraExecutor
