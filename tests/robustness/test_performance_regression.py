"""Performance regression tests.

These tests ensure that performance characteristics are maintained and
detect performance regressions in critical code paths.
"""

import asyncio
import os
import statistics
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment,unused-ignore]

from flujo.application.core.state_serializer import StateSerializer
from flujo.domain.models import ConversationRole, ConversationTurn, PipelineContext, StepResult
from tests.test_types.fixtures import create_test_step, create_test_step_result
from tests.test_types.fakes import FakeAgent
from tests.test_types.mocks import create_mock_executor_core
from tests.robustness.baseline_manager import get_baseline_manager

pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
]
if psutil is None:
    pytestmark.append(pytest.mark.skip(reason="psutil not available"))


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
            step = create_test_step(name="test_step", agent=FakeAgent())
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
            steps = [create_test_step(name=f"step_{i}", agent=FakeAgent()) for i in range(10)]
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

        def serialize_object():
            result = create_test_step_result(
                name="test",
                output={"data": "test" * 100},  # Larger object
                success=True,
            )
            return result.model_dump(mode="json")

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
            step_outputs={
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
        """Benchmark: Compare concurrent vs sequential execution performance.

        This is a BENCHMARK test - it logs performance metrics for tracking but
        does NOT assert on speedup. Performance assertions are inherently flaky
        in CI due to environment variance, task scheduling overhead, and system load.

        The test verifies CORRECTNESS (all operations complete with valid results)
        and logs detailed metrics for human review and trend analysis.
        """

        def get_system_metrics() -> dict[str, Any]:
            """Collect current system load metrics."""
            metrics = {}
            if psutil is not None:
                try:
                    process = psutil.Process(os.getpid())
                    metrics["cpu_percent"] = process.cpu_percent(interval=0.1)
                    metrics["memory_mb"] = process.memory_info().rss / 1024 / 1024
                    metrics["system_cpu_percent"] = psutil.cpu_percent(interval=0.1)
                    metrics["system_memory_percent"] = psutil.virtual_memory().percent
                except Exception as e:
                    metrics["error"] = str(e)
            return metrics

        async def execute_concurrent_steps():
            # --- Measure EXECUTOR INITIALIZATION TIME ---
            init_start = time.perf_counter()
            executor = create_mock_executor_core()
            init_time = (time.perf_counter() - init_start) * 1000  # ms

            num_operations = 10
            num_runs = 3  # Multiple runs for statistical analysis

            async def execute_single():
                step = create_test_step(name="concurrent_step", agent=FakeAgent())
                data = {"input": "concurrent_test"}
                await asyncio.sleep(0.001)  # 1ms simulated work
                return await executor.execute(step, data)

            # Collect metrics across multiple runs
            sequential_times: list[float] = []
            concurrent_times: list[float] = []
            speedups: list[float] = []
            system_metrics_list: list[dict[str, Any]] = []

            for run_idx in range(num_runs):
                # Collect system metrics before each run
                system_metrics = get_system_metrics()
                system_metrics_list.append(system_metrics)

                # --- Measure SEQUENTIAL execution with per-operation timing ---
                sequential_start = time.perf_counter()
                sequential_results = []
                sequential_operation_times: list[float] = []
                for op_idx in range(num_operations):
                    op_start = time.perf_counter()
                    result = await execute_single()
                    op_time = (time.perf_counter() - op_start) * 1000  # ms
                    sequential_operation_times.append(op_time)
                    sequential_results.append(result)
                sequential_time = (time.perf_counter() - sequential_start) * 1000  # ms

                # --- Measure CONCURRENT execution with task scheduling overhead ---
                # Measure task creation overhead
                task_creation_start = time.perf_counter()
                tasks = [execute_single() for _ in range(num_operations)]
                task_creation_time = (time.perf_counter() - task_creation_start) * 1000  # ms

                # Measure actual concurrent execution
                concurrent_start = time.perf_counter()
                concurrent_results = await asyncio.gather(*tasks)
                concurrent_time = (time.perf_counter() - concurrent_start) * 1000  # ms

                # --- Validate correctness (NOT performance) ---
                assert len(sequential_results) == num_operations, "Sequential: not all completed"
                assert len(concurrent_results) == num_operations, "Concurrent: not all completed"
                assert all(isinstance(r, StepResult) for r in sequential_results), (
                    "Invalid sequential results"
                )
                assert all(isinstance(r, StepResult) for r in concurrent_results), (
                    "Invalid concurrent results"
                )

                # Calculate speedup for logging
                speedup = sequential_time / concurrent_time if concurrent_time > 0 else float("inf")
                sequential_times.append(sequential_time)
                concurrent_times.append(concurrent_time)
                speedups.append(speedup)

                # Detailed logging for this run
                print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
                print(f"  Sequential: {sequential_time:.2f}ms")
                print(
                    f"    Per-operation: min={min(sequential_operation_times):.2f}ms, "
                    f"max={max(sequential_operation_times):.2f}ms, "
                    f"mean={statistics.mean(sequential_operation_times):.2f}ms"
                )
                print(f"  Concurrent: {concurrent_time:.2f}ms")
                print(f"    Task creation overhead: {task_creation_time:.2f}ms")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  System metrics: {system_metrics}")

            # --- Statistical Analysis ---
            mean_sequential = statistics.mean(sequential_times)
            mean_concurrent = statistics.mean(concurrent_times)
            mean_speedup = statistics.mean(speedups)
            median_speedup = statistics.median(speedups)
            actual_min_speedup = min(speedups)
            max_speedup = max(speedups)

            # Calculate standard deviation if we have enough samples
            speedup_std = statistics.stdev(speedups) if len(speedups) > 1 else 0.0

            # --- Comprehensive Benchmark Output ---
            print(f"\n{'=' * 60}")
            print("BENCHMARK: CONCURRENT EXECUTION PERFORMANCE")
            print(f"{'=' * 60}")
            print(f"Executor initialization time: {init_time:.2f}ms")
            print(f"\nSequential Execution (across {num_runs} runs):")
            print(f"  Mean: {mean_sequential:.2f}ms")
            print(f"  Min: {min(sequential_times):.2f}ms")
            print(f"  Max: {max(sequential_times):.2f}ms")
            print(f"\nConcurrent Execution (across {num_runs} runs):")
            print(f"  Mean: {mean_concurrent:.2f}ms")
            print(f"  Min: {min(concurrent_times):.2f}ms")
            print(f"  Max: {max(concurrent_times):.2f}ms")
            print("\nSpeedup Analysis:")
            print(f"  Mean: {mean_speedup:.2f}x")
            print(f"  Median: {median_speedup:.2f}x")
            print(f"  Min: {actual_min_speedup:.2f}x")
            print(f"  Max: {max_speedup:.2f}x")
            if speedup_std > 0:
                print(f"  Std Dev: {speedup_std:.2f}x")
            print("\nSystem Load (across runs):")
            if system_metrics_list:
                avg_cpu = statistics.mean([m.get("cpu_percent", 0) for m in system_metrics_list])
                avg_mem = statistics.mean([m.get("memory_mb", 0) for m in system_metrics_list])
                avg_sys_cpu = statistics.mean(
                    [m.get("system_cpu_percent", 0) for m in system_metrics_list]
                )
                avg_sys_mem = statistics.mean(
                    [m.get("system_memory_percent", 0) for m in system_metrics_list]
                )
                print(f"  Process CPU: {avg_cpu:.1f}%")
                print(f"  Process Memory: {avg_mem:.1f}MB")
                print(f"  System CPU: {avg_sys_cpu:.1f}%")
                print(f"  System Memory: {avg_sys_mem:.1f}%")
            print(f"{'=' * 60}")

            # NOTE: No performance assertion - this is a benchmark, not a gate.
            # The metrics above are logged for human review and trend analysis.
            # Speedup varies significantly based on CI environment load.

        asyncio.run(execute_concurrent_steps())

    @pytest.mark.asyncio
    async def test_caching_performance_improvement(self, baseline_manager):
        """Test that caching provides measurable performance improvement."""
        # Use stateful mock: first call miss (None), second call hit (StepResult)
        executor = create_mock_executor_core(cache_hit=False)
        step = create_test_step(name="cached_step", agent=FakeAgent())

        cached_result = StepResult(
            name="cached_step",
            output={"cached": "result"},
            success=True,
            metadata_={"cache_hit": True},
        )
        executor._cache_backend.get = AsyncMock(side_effect=[None, cached_result])

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


class TestScalabilityRegression:
    """Test suite for scalability regression detection."""

    def test_large_pipeline_performance(self):
        """Test that large pipelines scale reasonably."""
        from flujo.domain.dsl.pipeline import Pipeline

        # Create a large pipeline
        num_steps = 50
        steps = [create_test_step(name=f"step_{i}", agent=FakeAgent()) for i in range(num_steps)]
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

        async def run_high_concurrency_test():
            executor = create_mock_executor_core()

            # Warm up executor to avoid one-time initialization costs
            await executor.execute(
                create_test_step(name="warmup", agent=FakeAgent()), {"input": "warmup"}
            )

            num_operations = 20  # Use 20 operations for reliable measurement

            steps = [
                create_test_step(name=f"concurrency_test_{i}", agent=FakeAgent())
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
            print("\nHigh Concurrency Performance:")
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
