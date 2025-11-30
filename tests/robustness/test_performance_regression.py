"""Performance regression tests.

These tests ensure that performance characteristics are maintained and
detect performance regressions in critical code paths.
"""
# ruff: noqa

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]

import asyncio
import time
import psutil
import os
from typing import Dict, List, Any, Optional
import pytest
from unittest.mock import AsyncMock

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult
from tests.test_types.mocks import create_mock_executor_core
from tests.robustness.baseline_manager import get_baseline_manager, measure_and_check_regression


class TestPerformanceRegression:
    """Test suite for performance regression detection."""

    @pytest.fixture
    def baseline_manager(self):
        """Performance baseline manager."""
        return get_baseline_manager()

    @pytest.fixture
    def baseline_thresholds(self) -> Dict[str, float]:
        """Provide default performance thresholds for robustness tests."""
        # Generous defaults to avoid flaky regressions; can be tightened as needed.
        return {
            "pipeline_creation": 50.0,
            "context_isolation": 50.0,
            "serialization": 50.0,
            "memory_overhead": 100.0,
        }

    def measure_execution_time(self, func, *args, **kwargs) -> tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time

    async def measure_async_execution_time(self, coro) -> tuple[Any, float]:
        """Measure execution time of an async function."""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time

    def test_step_execution_performance(self, baseline_manager):
        """Test that step execution performance stays within acceptable bounds."""
        executor = create_mock_executor_core()

        async def execute_step():
            step = Step(name="test_step", agent=AsyncMock())
            data = {"input": "test"}
            result = await executor.execute(step, data)
            return result

        async def run_performance_test():
            # Warm up
            await execute_step()

            # Measure performance
            results = []
            for _ in range(10):
                _, execution_time = await self.measure_async_execution_time(execute_step())
                results.append(execution_time)

            avg_time = sum(results) / len(results)

            # Add measurement to baseline and check for regression
            baseline_manager.add_measurement("step_execution", avg_time)
            is_regression, message = baseline_manager.check_regression("step_execution", avg_time)

            # Log performance data
            print(f"Step execution: {avg_time:.2f}ms ({message})")

            assert not is_regression, f"Performance regression detected: {message}"

        asyncio.run(run_performance_test())

    def test_pipeline_creation_performance(self, baseline_thresholds: Dict[str, float]):
        """Test that pipeline creation performance stays within bounds."""
        from flujo.domain.dsl.pipeline import Pipeline

        def create_test_pipeline():
            steps = [Step(name=f"step_{i}", agent=AsyncMock()) for i in range(10)]
            return Pipeline(steps=steps)

        # Warm up
        create_test_pipeline()

        # Measure performance
        results = []
        for _ in range(10):
            _, execution_time = self.measure_execution_time(create_test_pipeline)
            results.append(execution_time)

        avg_time = sum(results) / len(results)
        threshold = baseline_thresholds["pipeline_creation"]

        assert avg_time <= threshold, (
            f"Pipeline creation time {avg_time:.2f}ms exceeds threshold {threshold}ms"
        )

    def test_context_isolation_performance(self, baseline_thresholds: Dict[str, float]):
        """Test that context isolation performance stays within bounds."""
        from flujo.application.core.context_manager import ContextManager
        from flujo.domain.models import PipelineContext

        def isolate_context():
            context = PipelineContext()
            return ContextManager.isolate(context)

        # Warm up
        isolate_context()

        # Measure performance
        results = []
        for _ in range(100):
            _, execution_time = self.measure_execution_time(isolate_context)
            results.append(execution_time)

        avg_time = sum(results) / len(results)
        threshold = baseline_thresholds["context_isolation"]

        assert avg_time <= threshold, (
            f"Context isolation time {avg_time:.2f}ms exceeds threshold {threshold}ms"
        )

    def test_serialization_performance(self, baseline_thresholds: Dict[str, float]):
        """Test that serialization performance stays within bounds."""
        from flujo.utils.serialization import safe_serialize
        from flujo.domain.models import StepResult

        def serialize_object():
            result = StepResult(
                name="test",
                output={"data": "test" * 100},  # Larger object
                success=True,
            )
            return safe_serialize(result)

        # Warm up
        serialize_object()

        # Measure performance
        results = []
        for _ in range(10):
            _, execution_time = self.measure_execution_time(serialize_object)
            results.append(execution_time)

        avg_time = sum(results) / len(results)
        threshold = baseline_thresholds["serialization"]

        assert avg_time <= threshold, (
            f"Serialization time {avg_time:.2f}ms exceeds threshold {threshold}ms"
        )

    def test_memory_overhead_monitoring(self, baseline_thresholds: Dict[str, float]):
        """Test that memory overhead stays within acceptable bounds."""
        import gc

        def get_memory_usage():
            """Get current memory usage in MB."""
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        # Force garbage collection
        gc.collect()

        # Baseline memory
        baseline_memory = get_memory_usage()

        # Perform operations that should not cause significant memory growth
        results = []
        for i in range(100):
            result = StepResult(name=f"test_{i}", output={"data": f"value_{i}"}, success=True)
            results.append(result)

            if i % 10 == 0:
                current_memory = get_memory_usage()
                memory_growth = ((current_memory - baseline_memory) / baseline_memory) * 100
                threshold = baseline_thresholds["memory_overhead"]

                assert memory_growth <= threshold, (
                    f"Memory growth {memory_growth:.1f}% exceeds threshold {threshold}% "
                    f"at iteration {i} (baseline: {baseline_memory:.1f}MB, current: {current_memory:.1f}MB)"
                )

        # Cleanup
        del results
        gc.collect()

    def test_concurrent_execution_performance(self):
        """Test that concurrent execution doesn't degrade performance significantly."""
        import asyncio

        async def execute_concurrent_steps():
            executor = create_mock_executor_core()

            async def execute_single():
                step = Step(name="concurrent_step", agent=AsyncMock())
                data = {"input": "concurrent_test"}
                return await executor.execute(step, data)

            # Execute 10 concurrent operations
            tasks = [execute_single() for _ in range(10)]
            start_time = time.perf_counter()

            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000  # ms

            # Sequential would take ~500ms (50ms * 10), concurrent should be much faster
            # Allow some overhead but ensure reasonable speedup
            assert total_time < 200, (
                f"Concurrent execution took {total_time:.1f}ms, expected < 200ms"
            )
            assert len(results) == 10, "Not all concurrent operations completed"
            assert all(isinstance(r, StepResult) for r in results), "Invalid results"

        asyncio.run(execute_concurrent_steps())

    def test_caching_performance_improvement(self, baseline_manager):
        """Test that caching provides measurable performance improvement."""
        executor = create_mock_executor_core(cache_hit=True)
        step = Step(name="cached_step", agent=AsyncMock())

        async def run_cached_test():
            # First execution (cache miss)
            _, first_time = await self.measure_async_execution_time(
                executor.execute(step, {"input": "test"})
            )

            # Second execution (cache hit)
            _, second_time = await self.measure_async_execution_time(
                executor.execute(step, {"input": "test"})
            )

            # Calculate improvement ratio
            improvement_ratio = first_time / second_time if second_time > 0 else float("inf")

            # Add measurement to baseline and check for regression
            baseline_manager.add_measurement("cache_improvement_ratio", improvement_ratio)
            is_regression, message = baseline_manager.check_regression(
                "cache_improvement_ratio", improvement_ratio
            )

            # Log performance data
            print(f"Cache improvement: {improvement_ratio:.2f}x ({message})")

            assert not is_regression, f"Cache performance regression detected: {message}"

        asyncio.run(run_cached_test())


class TestScalabilityRegression:
    """Test suite for scalability regression detection."""

    def test_large_pipeline_performance(self):
        """Test that large pipelines scale reasonably."""
        from flujo.domain.dsl.pipeline import Pipeline

        # Create a large pipeline
        num_steps = 50
        steps = [Step(name=f"step_{i}", agent=AsyncMock()) for i in range(num_steps)]
        pipeline = Pipeline(steps=steps)

        # Measure pipeline creation time
        def create_large_pipeline():
            return Pipeline(steps=steps)

        _, creation_time = TestPerformanceRegression().measure_execution_time(create_large_pipeline)

        # Large pipeline creation should still be reasonable (< 100ms)
        assert creation_time < 100, f"Large pipeline creation took {creation_time:.2f}ms"

        # Memory usage should be reasonable
        import sys

        pipeline_size = sys.getsizeof(pipeline)
        assert pipeline_size < 1024 * 1024, f"Pipeline memory usage {pipeline_size} bytes too high"

    def test_high_concurrency_handling(self):
        """Test that the system handles high concurrency without degradation."""
        import asyncio

        async def run_high_concurrency_test():
            executor = create_mock_executor_core()

            # Warm up executor to avoid one-time initialization costs impacting timing
            await executor.execute(Step(name="warmup", agent=AsyncMock()), {"input": "warmup"})

            steps = [Step(name=f"concurrency_test_{i}", agent=AsyncMock()) for i in range(100)]

            async def execute_with_delay(step: Step):
                await asyncio.sleep(0.0005)  # Small delay to simulate work
                return await executor.execute(step, {"input": "test"})

            # Run 100 concurrent operations
            num_concurrent = 100
            tasks = [execute_with_delay(step) for step in steps[:num_concurrent]]

            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            total_time = (end_time - start_time) * 1000  # ms

            # With 1ms delay per operation, sequential would be ~100ms
            # Concurrent should be much faster but allow for test overhead
            assert total_time < 150, (
                f"High concurrency test took {total_time:.1f}ms, expected < 150ms"
            )
            assert len(results) == num_concurrent, "Not all operations completed"
            assert all(isinstance(r, StepResult) for r in results), "Invalid results"

        asyncio.run(run_high_concurrency_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
