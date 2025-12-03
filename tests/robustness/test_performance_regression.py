"""Performance regression tests.

These tests ensure that performance characteristics are maintained and
detect performance regressions in critical code paths.
"""
# ruff: noqa

import pytest

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment,unused-ignore]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]
if psutil is None:
    pytestmark.append(pytest.mark.skip(reason="psutil not available"))

import asyncio
import time
import os
from typing import Any
from unittest.mock import AsyncMock

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.state_serializer import StateSerializer
from flujo.domain.models import ConversationRole, ConversationTurn, PipelineContext, StepResult
from tests.test_types.fixtures import create_test_step, create_test_step_result
from tests.test_types.mocks import create_mock_executor_core
from tests.robustness.baseline_manager import get_baseline_manager, measure_and_check_regression


class TestPerformanceRegression:
    """Test suite for performance regression detection."""

    @pytest.fixture
    def baseline_manager(self):
        """Performance baseline manager."""
        return get_baseline_manager()

    @pytest.fixture
    def baseline_thresholds(self) -> dict[str, float]:
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
            step = create_test_step(name="test_step", agent=AsyncMock())
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

    def test_pipeline_creation_performance(self, baseline_thresholds: dict[str, float]):
        """Test that pipeline creation performance stays within bounds."""
        from flujo.domain.dsl.pipeline import Pipeline

        def create_test_pipeline():
            steps = [create_test_step(name=f"step_{i}", agent=AsyncMock()) for i in range(10)]
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

    def test_context_isolation_performance(self, baseline_thresholds: dict[str, float]):
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

    def test_serialization_performance(self, baseline_thresholds: dict[str, float]):
        """Test that serialization performance stays within bounds."""
        from flujo.utils.serialization import safe_serialize
        from flujo.domain.models import StepResult

        def serialize_object():
            result = create_test_step_result(
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

    def test_context_hash_performance(self):
        """Ensure context hashing stays fast for large contexts."""
        serializer: StateSerializer[PipelineContext] = StateSerializer()

        conversation = [
            ConversationTurn(
                role=ConversationRole.user if i % 2 == 0 else ConversationRole.assistant,
                content=f"message {i} {'x' * 10}",
            )
            for i in range(200)
        ]
        context = PipelineContext(
            initial_prompt="performance check",
            conversation_history=conversation,
            scratchpad={
                "metrics": [{"index": i, "values": list(range(15))} for i in range(50)],
                "notes": "n" * 256,
            },
        )

        # Warm up to populate caches
        serializer.compute_context_hash(context)

        start_time = time.perf_counter()
        for _ in range(5):
            serializer.compute_context_hash(context)
        total_time = (time.perf_counter() - start_time) * 1000  # ms

        assert total_time < 75, (
            f"Context hashing took {total_time:.1f}ms for 5 runs, expected < 75ms"
        )

    def test_memory_overhead_monitoring(self, baseline_thresholds: dict[str, float]):
        """Test that memory overhead stays within acceptable bounds."""
        import gc

        def get_memory_usage():
            """Get current memory usage in MB."""
            if psutil is None:
                pytest.skip("psutil not available")
            try:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024  # MB
            except (KeyError, OSError, psutil.Error) as e:
                pytest.skip(f"psutil not working in this environment: {e}")

        # Force garbage collection
        gc.collect()

        # Baseline memory
        baseline_memory = get_memory_usage()

        # Perform operations that should not cause significant memory growth
        results = []
        for i in range(100):
            result = create_test_step_result(
                name=f"test_{i}", output={"data": f"value_{i}"}, success=True
            )
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
        """Test that concurrent execution provides speedup over sequential.

        This test uses RELATIVE performance measurement instead of absolute thresholds.
        By comparing concurrent vs sequential execution in the same environment,
        we get an environment-independent test that works identically in local and CI.
        """
        import asyncio

        async def execute_concurrent_steps():
            executor = create_mock_executor_core()

            num_operations = 10

            async def execute_single():
                step = create_test_step(name="concurrent_step", agent=AsyncMock())
                data = {"input": "concurrent_test"}
                await asyncio.sleep(0.001)  # 1ms simulated work
                return await executor.execute(step, data)

            # --- Measure SEQUENTIAL execution ---
            sequential_start = time.perf_counter()
            sequential_results = []
            for _ in range(num_operations):
                result = await execute_single()
                sequential_results.append(result)
            sequential_time = (time.perf_counter() - sequential_start) * 1000  # ms

            # --- Measure CONCURRENT execution ---
            tasks = [execute_single() for _ in range(num_operations)]
            concurrent_start = time.perf_counter()
            concurrent_results = await asyncio.gather(*tasks)
            concurrent_time = (time.perf_counter() - concurrent_start) * 1000  # ms

            # --- Validate correctness ---
            assert len(sequential_results) == num_operations, "Sequential: not all completed"
            assert len(concurrent_results) == num_operations, "Concurrent: not all completed"
            assert all(isinstance(r, StepResult) for r in sequential_results), (
                "Invalid sequential results"
            )
            assert all(isinstance(r, StepResult) for r in concurrent_results), (
                "Invalid concurrent results"
            )

            # --- Validate RELATIVE performance ---
            # Concurrent should be faster than sequential.
            # With 10 operations and 1ms delay each:
            # - Sequential: ~10ms (delays) + overhead
            # - Concurrent: ~1ms (delays in parallel) + overhead
            #
            # We require at least 1.1x speedup as a sanity check.
            # Note: In CI environments, overhead dominates and speedup is lower.
            # Local typically sees 1.5-2x, CI sees 1.1-1.4x.
            # The key validation is that concurrent execution is faster than sequential.
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else float("inf")
            min_speedup = 1.1

            print(f"\nConcurrent Execution Performance:")
            print(f"  Sequential: {sequential_time:.1f}ms")
            print(f"  Concurrent: {concurrent_time:.1f}ms")
            print(f"  Speedup: {speedup:.2f}x")

            assert speedup >= min_speedup, (
                f"Concurrent execution should be at least {min_speedup}x faster than sequential. "
                f"Got {speedup:.2f}x (sequential={sequential_time:.1f}ms, concurrent={concurrent_time:.1f}ms)"
            )

        asyncio.run(execute_concurrent_steps())

    def test_caching_performance_improvement(self, baseline_manager):
        """Test that caching provides measurable performance improvement."""
        executor = create_mock_executor_core(cache_hit=True)
        step = create_test_step(name="cached_step", agent=AsyncMock())

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
        steps = [create_test_step(name=f"step_{i}", agent=AsyncMock()) for i in range(num_steps)]
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
        """Test that the system handles high concurrency without degradation.

        This test uses RELATIVE performance measurement (speedup over sequential)
        rather than absolute time thresholds. This makes the test environment-independent:
        - Local machine: faster absolute times, same relative speedup
        - CI environment: slower absolute times, same relative speedup

        The key invariant is: concurrent execution should be faster than sequential.
        """
        import asyncio

        async def run_high_concurrency_test():
            executor = create_mock_executor_core()

            # Warm up executor to avoid one-time initialization costs
            await executor.execute(
                create_test_step(name="warmup", agent=AsyncMock()), {"input": "warmup"}
            )

            num_operations = 20  # Use 20 operations for reliable measurement

            steps = [
                create_test_step(name=f"concurrency_test_{i}", agent=AsyncMock())
                for i in range(num_operations)
            ]

            async def execute_with_delay(step: Any):
                await asyncio.sleep(0.001)  # 1ms delay to simulate work
                return await executor.execute(step, {"input": "test"})

            # --- Measure SEQUENTIAL execution ---
            sequential_start = time.perf_counter()
            sequential_results = []
            for step in steps:
                result = await execute_with_delay(step)
                sequential_results.append(result)
            sequential_time = (time.perf_counter() - sequential_start) * 1000  # ms

            # --- Measure CONCURRENT execution ---
            tasks = [execute_with_delay(step) for step in steps]
            concurrent_start = time.perf_counter()
            concurrent_results = await asyncio.gather(*tasks)
            concurrent_time = (time.perf_counter() - concurrent_start) * 1000  # ms

            # --- Validate correctness ---
            assert len(sequential_results) == num_operations, "Sequential: not all completed"
            assert len(concurrent_results) == num_operations, "Concurrent: not all completed"
            assert all(isinstance(r, StepResult) for r in sequential_results), (
                "Invalid sequential results"
            )
            assert all(isinstance(r, StepResult) for r in concurrent_results), (
                "Invalid concurrent results"
            )

            # --- Validate RELATIVE performance (environment-independent) ---
            # Concurrent execution should provide speedup over sequential.
            # With 20 operations and 1ms delay each:
            # - Sequential: ~20ms (delays) + overhead
            # - Concurrent: ~1ms (delays run in parallel) + overhead
            #
            # We require at least 1.5x speedup as a sanity check.
            # This is conservative because:
            # - True parallelism would give ~20x speedup on the delays alone
            # - But executor overhead and asyncio scheduling reduce this
            # - 1.5x ensures we're not accidentally running sequentially

            speedup = sequential_time / concurrent_time if concurrent_time > 0 else float("inf")
            min_speedup = 1.5

            # Log performance for debugging (not part of assertion)
            print(f"\nHigh Concurrency Performance:")
            print(f"  Sequential: {sequential_time:.1f}ms")
            print(f"  Concurrent: {concurrent_time:.1f}ms")
            print(f"  Speedup: {speedup:.2f}x")

            assert speedup >= min_speedup, (
                f"Concurrent execution should be at least {min_speedup}x faster than sequential. "
                f"Got {speedup:.2f}x speedup (sequential={sequential_time:.1f}ms, "
                f"concurrent={concurrent_time:.1f}ms)"
            )

        asyncio.run(run_high_concurrency_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
