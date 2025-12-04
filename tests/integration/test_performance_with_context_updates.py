"""
Integration tests for Performance Testing + Context Updates feature combination.

This tests the critical combination of performance testing with context-updating steps,
which could reveal bugs in performance bottlenecks and memory issues with large context objects.
"""

import pytest
import time
import asyncio
from typing import Any, List
from flujo import step, Step
from flujo.domain.models import PipelineContext
from flujo.domain.dsl.step import MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class PerformanceContext(PipelineContext):
    """Context for testing performance with context updates."""

    initial_prompt: str = "test"
    operation_count: int = 0
    large_data: str = "x" * 2000  # 2KB of data (reduced from 10KB for faster tests)
    nested_data: JSONObject = {"deep": {"nested": {"data": "x" * 1000}}}
    performance_metrics: dict[str, float] = {}
    memory_usage: List[int] = []
    execution_times: List[float] = []
    context_updates: int = 0
    large_list: List[Any] = ["item"] * 100  # 100 items, can be strings or dicts (reduced from 1000)
    complex_object: JSONObject = {
        "nested": {"deep": {"structure": {"with": {"many": {"levels": "data"}}}}},
        "arrays": [{"item": i, "data": "x" * 100} for i in range(10)],  # Reduced from 100 to 10
        "strings": ["string"] * 50,
    }


@step(updates_context=True)
async def performance_step(data: Any, *, context: PerformanceContext) -> JSONObject:
    """Step that updates context with performance metrics."""
    start_time = time.time()
    context.operation_count += 1
    context.context_updates += 1

    # Simulate some work (reduced sleep for faster tests)
    await asyncio.sleep(0.001)

    execution_time = time.time() - start_time
    context.execution_times.append(execution_time)
    context.performance_metrics[f"operation_{context.operation_count}"] = execution_time

    return {"operation_count": context.operation_count, "context_updates": context.context_updates}


@step(updates_context=True)
async def large_context_step(data: Any, *, context: PerformanceContext) -> JSONObject:
    """Step that works with large context data."""
    context.operation_count += 1
    context.context_updates += 1

    # Update large data structures
    context.large_data += f"_update_{context.operation_count}"
    context.large_list.append(f"new_item_{context.operation_count}")
    context.complex_object["arrays"].append(
        {"item": len(context.complex_object["arrays"]), "data": "x" * 200}
    )

    return {"operation_count": context.operation_count, "context_updates": context.context_updates}


@step(updates_context=True)
async def high_frequency_step(data: Any, *, context: PerformanceContext) -> JSONObject:
    """Step that performs high-frequency context updates."""
    context.operation_count += 1
    context.context_updates += 1

    # Rapid context updates
    for i in range(10):
        context.performance_metrics[f"rapid_{context.operation_count}_{i}"] = time.time()
        context.memory_usage.append(len(str(context.large_data)))

    return {"operation_count": context.operation_count, "context_updates": context.context_updates}


@step(updates_context=True)
async def memory_intensive_step(data: Any, *, context: PerformanceContext) -> JSONObject:
    """Step that performs memory-intensive operations."""
    context.operation_count += 1
    context.context_updates += 1

    # Create large data structures
    large_string = "x" * 2000  # 2KB (reduced from 10KB for faster tests)
    large_list = [{"data": large_string, "index": i} for i in range(10)]  # Reduced from 100 to 10

    # Update context with large data
    context.large_data = large_string
    context.large_list = large_list
    context.complex_object["large_arrays"] = large_list

    return {"operation_count": context.operation_count, "context_updates": context.context_updates}


@pytest.mark.asyncio
async def test_performance_with_context_updates_basic():
    """Test basic performance operations with context updates."""

    pipeline = performance_step
    runner = create_test_flujo(pipeline, context_model=PerformanceContext, persist_state=False)

    # Warmup run to avoid cold-start effects
    await gather_result(runner, "warmup")

    # Run multiple times to test performance
    start_time = time.time()
    results = []

    for i in range(10):
        result = await gather_result(runner, f"test_input_{i}")
        results.append(result)

    total_time = time.time() - start_time
    print(f"Basic performance test: {total_time:.2f}s for 10 runs ({total_time / 10:.3f}s per run)")

    # Verify all runs completed successfully
    for result in results:
        assert result.step_history[-1].success is True
        assert result.final_pipeline_context.operation_count == 1  # Each run gets fresh context
        assert result.final_pipeline_context.context_updates == 1

    # Sanity check: 10 runs should complete within 30s (major regression detection)
    assert total_time < 30.0, f"Performance test too slow: {total_time:.2f}s (major regression)"

    # Verify context updates are working (check last result)
    final_result = results[-1]
    assert final_result.final_pipeline_context.operation_count == 1
    assert final_result.final_pipeline_context.context_updates == 1
    assert len(final_result.final_pipeline_context.execution_times) == 1


@pytest.mark.asyncio
async def test_performance_with_context_updates_large_context():
    """Test performance with large context objects."""

    pipeline = large_context_step
    runner = create_test_flujo(pipeline, context_model=PerformanceContext, persist_state=False)

    # Run with large context data
    start_time = time.time()
    result = await gather_result(runner, "large_context_test")
    execution_time = time.time() - start_time

    # Verify successful execution
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.operation_count == 1
    assert result.final_pipeline_context.context_updates == 1

    # Verify large data was updated
    # large_data starts at 2000 chars, large_context_step appends "_update_1" (8 chars) = 2008 total
    assert len(result.final_pipeline_context.large_data) > 2000
    # large_list starts at 100 items, large_context_step appends 1 item = 101 total
    assert len(result.final_pipeline_context.large_list) > 100
    # complex_object["arrays"] starts at 10 items, large_context_step appends 1 item = 11 total
    assert len(result.final_pipeline_context.complex_object["arrays"]) > 10

    print(f"Large context test: {execution_time:.2f}s")
    # Sanity check: large context operation should complete within 30s (major regression detection)
    assert execution_time < 30.0, (
        f"Large context test too slow: {execution_time:.2f}s (major regression)"
    )


@pytest.mark.asyncio
async def test_performance_with_context_updates_high_frequency():
    """Test performance with high-frequency context updates."""

    pipeline = high_frequency_step
    runner = create_test_flujo(pipeline, context_model=PerformanceContext, persist_state=False)

    # Run multiple times to test high-frequency updates
    start_time = time.time()
    results = []

    for i in range(20):
        result = await gather_result(runner, f"high_freq_input_{i}")
        results.append(result)

    total_time = time.time() - start_time

    # Verify all runs completed successfully
    for result in results:
        assert result.step_history[-1].success is True
        assert result.final_pipeline_context.operation_count == 1  # Each run gets fresh context
        assert result.final_pipeline_context.context_updates == 1

    print(f"High frequency test: {total_time:.2f}s for 20 runs ({total_time / 20:.3f}s per run)")
    # Sanity check: 20 high-frequency runs should complete within 60s (major regression detection)
    assert total_time < 60.0, f"High frequency test too slow: {total_time:.2f}s (major regression)"

    # Verify high-frequency updates (check last result)
    final_result = results[-1]
    assert final_result.final_pipeline_context.operation_count == 1
    assert final_result.final_pipeline_context.context_updates == 1
    assert (
        len(final_result.final_pipeline_context.performance_metrics) >= 10
    )  # 1 * 10 rapid updates


@pytest.mark.asyncio
async def test_performance_with_context_updates_memory_intensive():
    """Test performance with memory-intensive operations."""

    pipeline = memory_intensive_step
    runner = create_test_flujo(pipeline, context_model=PerformanceContext, persist_state=False)

    # Run memory-intensive operation
    start_time = time.time()
    result = await gather_result(runner, "memory_intensive_test")
    execution_time = time.time() - start_time

    # Verify successful execution
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.operation_count == 1
    assert result.final_pipeline_context.context_updates == 1

    # Verify large data was created
    # memory_intensive_step sets large_data to "x" * 2000 = 2000 chars
    assert len(result.final_pipeline_context.large_data) >= 2000
    # memory_intensive_step sets large_list to 10 items
    assert len(result.final_pipeline_context.large_list) >= 10
    assert "large_arrays" in result.final_pipeline_context.complex_object

    print(f"Memory intensive test: {execution_time:.2f}s")
    # Sanity check: memory intensive operation should complete within 30s (major regression detection)
    assert execution_time < 30.0, (
        f"Memory intensive test too slow: {execution_time:.2f}s (major regression)"
    )


@pytest.mark.asyncio
async def test_performance_with_context_updates_parallel():
    """Test performance with parallel context updates."""

    # Create parallel pipeline with context-updating steps
    parallel_pipeline = Step.parallel(
        "parallel_performance",
        {
            "perf1": performance_step,
            "perf2": performance_step,
            "perf3": performance_step,
        },
        merge_strategy=MergeStrategy.OVERWRITE,
        context_include_keys=["operation_count", "context_updates"],
    )

    runner = create_test_flujo(
        parallel_pipeline, context_model=PerformanceContext, persist_state=False
    )

    # Run parallel performance test
    start_time = time.time()
    result = await gather_result(runner, "parallel_performance_test")
    execution_time = time.time() - start_time

    # Verify successful execution
    assert result.step_history[-1].success is True
    assert len(result.step_history) == 1  # Parallel steps are aggregated into one result

    # Verify parallel context updates (should be aggregated, but only one operation per branch)
    assert result.final_pipeline_context.operation_count >= 1  # At least one operation
    assert result.final_pipeline_context.context_updates >= 1  # At least one context update

    print(f"Parallel performance test: {execution_time:.2f}s")
    # Sanity check: parallel test should complete within 30s (major regression detection)
    assert execution_time < 30.0, (
        f"Parallel performance test too slow: {execution_time:.2f}s (major regression)"
    )


@pytest.mark.asyncio
async def test_performance_with_context_updates_complex_pipeline():
    """Test performance with complex pipeline involving multiple context updates."""

    # Create complex pipeline with multiple context-updating steps
    complex_pipeline = Pipeline(
        steps=[
            performance_step,
            large_context_step,
            high_frequency_step,
            memory_intensive_step,
        ]
    )

    runner = create_test_flujo(
        complex_pipeline, context_model=PerformanceContext, persist_state=False
    )

    # Run complex pipeline
    start_time = time.time()
    result = await gather_result(runner, "complex_performance_test")
    execution_time = time.time() - start_time

    # Verify successful execution
    assert result.step_history[-1].success is True
    assert len(result.step_history) == 4

    # Verify all steps executed and updated context
    assert result.final_pipeline_context.operation_count == 4
    assert result.final_pipeline_context.context_updates == 4

    # Verify large data was processed
    # memory_intensive_step (last step) sets large_data to "x" * 2000 = 2000 chars
    assert len(result.final_pipeline_context.large_data) >= 2000
    # memory_intensive_step sets large_list to 10 items
    assert len(result.final_pipeline_context.large_list) >= 10
    assert len(result.final_pipeline_context.performance_metrics) > 0

    print(f"Complex pipeline test: {execution_time:.2f}s")
    # Sanity check: complex pipeline should complete within 60s (major regression detection)
    assert execution_time < 60.0, (
        f"Complex pipeline test too slow: {execution_time:.2f}s (major regression)"
    )


@pytest.mark.asyncio
async def test_performance_with_context_updates_error_handling():
    """Test performance with error handling and context updates."""

    @step(updates_context=True)
    async def error_prone_step(data: Any, *, context: PerformanceContext) -> JSONObject:
        """Step that may fail but updates context."""
        context.operation_count += 1
        context.context_updates += 1

        if context.operation_count % 3 == 0:  # Fail every 3rd operation
            raise RuntimeError(f"Intentional failure at operation {context.operation_count}")

        return {
            "operation_count": context.operation_count,
            "context_updates": context.context_updates,
        }

    pipeline = error_prone_step
    runner = create_test_flujo(pipeline, context_model=PerformanceContext, persist_state=False)

    # Run multiple times to test error handling
    start_time = time.time()
    results = []

    for i in range(10):
        try:
            result = await gather_result(runner, f"error_test_{i}")
            results.append(result)
        except Exception:
            # Expected some failures
            pass

    total_time = time.time() - start_time

    # Verify some runs completed successfully (should be about 7 out of 10)
    assert len(results) >= 5, f"Expected at least 5 successful runs, got {len(results)}"

    print(f"Error handling test: {total_time:.2f}s for 10 runs")
    # Sanity check: error handling test should complete within 60s (major regression detection)
    assert total_time < 60.0, f"Error handling test too slow: {total_time:.2f}s (major regression)"

    # Verify context updates from successful runs
    if results:
        final_result = results[-1]
        assert final_result.final_pipeline_context.operation_count > 0
        assert final_result.final_pipeline_context.context_updates > 0
