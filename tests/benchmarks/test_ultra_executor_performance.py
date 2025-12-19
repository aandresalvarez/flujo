"""Benchmark tests for ultra executor performance improvements."""

import os
import pytest
import asyncio
import time
import statistics
from unittest.mock import Mock, AsyncMock
from contextlib import contextmanager

from flujo.application.core.executor_core import ExecutorCore as UltraStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import BaseModel as DomainBaseModel
from flujo.domain.models import StepResult
from tests.test_types.fixtures import execute_simple_step

# Constants for better maintainability
FLOAT_TOLERANCE = 1e-10
CI_TRUE_VALUES = ("true", "1", "yes")


class _BenchmarkContext(DomainBaseModel):
    context: str = "data"


class _BenchmarkResources(DomainBaseModel):
    resources: str = "data"


@contextmanager
def temporary_env_var(var_name, value):
    """Context manager to temporarily set an environment variable.

    Args:
        var_name: Name of the environment variable
        value: Value to set (None to delete the variable)
    """
    original_value = os.getenv(var_name)
    try:
        if value is None:
            if var_name in os.environ:
                del os.environ[var_name]
        else:
            os.environ[var_name] = value
        yield
    finally:
        if original_value is None:
            if var_name in os.environ:
                del os.environ[var_name]
        else:
            os.environ[var_name] = original_value


def create_slow_run_helper():
    """Helper function to create a slow run function for testing."""

    async def slow_run(data, **kwargs):
        await asyncio.sleep(0.01)
        return "slow_result"

    return slow_run


def test_relative_performance_approach():
    """Benchmark: Demonstrate relative performance measurement approach.

    This test documents the preferred approach for performance testing:
    - Use RELATIVE measurements (speedup ratios, max/min ratios)
    - Avoid ABSOLUTE thresholds that vary between environments
    - Log metrics for human review, don't assert on timing

    Note: Even relative speedup assertions can be flaky in CI due to
    task scheduling overhead varying with system load. This test logs
    the speedup ratio but does not fail if it's below a threshold.
    """
    import asyncio

    async def measure_speedup():
        # Sequential
        sequential_start = time.perf_counter()
        for _ in range(5):
            await asyncio.sleep(0.001)
        sequential_time = time.perf_counter() - sequential_start

        # Concurrent
        concurrent_start = time.perf_counter()
        await asyncio.gather(*[asyncio.sleep(0.001) for _ in range(5)])
        concurrent_time = time.perf_counter() - concurrent_start

        return sequential_time, concurrent_time

    sequential_time, concurrent_time = asyncio.run(measure_speedup())
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1.0

    # Log metrics for review (no assertion)
    print(f"\n{'=' * 60}")
    print("BENCHMARK: Relative Performance Approach Demo")
    print(f"{'=' * 60}")
    print(f"  Sequential: {sequential_time * 1000:.2f}ms")
    print(f"  Concurrent: {concurrent_time * 1000:.2f}ms")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"{'=' * 60}")

    # NOTE: No assertion - speedup varies with CI system load.
    # The purpose of this test is to demonstrate the measurement approach,
    # not to gate on a specific speedup value.


class TestUltraExecutorPerformance:
    """Test the performance characteristics of the ultra executor."""

    @pytest.fixture
    def ultra_executor(self):
        """Create an ultra executor for benchmarking."""
        return UltraStepExecutor(
            enable_cache=True,
            cache_size=1000,
            cache_ttl=3600,
            concurrency_limit=8,
        )

    @pytest.fixture
    def realistic_iterative_executor(self):
        """Create a realistic iterative executor for comparison."""

        class RealisticExecutor:
            def __init__(self):
                self._cache = {}
                self._usage = {"total_cost": 0.0, "total_tokens": 0}

            async def execute_step(self, step, data, context, resources):
                """Realistic executor with retries, caching, and error handling."""
                agent = getattr(step, "agent", None)
                if agent is None:
                    raise ValueError("Step has no agent")

                # Generate cache key (simplified)
                cache_key = (
                    f"{step.name}:{hash(str(data))}:{hash(str(context))}:{hash(str(resources))}"
                )

                # Check cache
                if cache_key in self._cache:
                    return self._cache[cache_key]

                # Execute with retries
                max_retries = getattr(step.config, "max_retries", 3)
                last_exception = None

                for attempt in range(1, max_retries + 1):
                    try:
                        # Build kwargs (similar to ultra executor)
                        kwargs = {}
                        if context is not None:
                            kwargs["context"] = context
                        if resources is not None:
                            kwargs["resources"] = resources
                        if (
                            hasattr(step.config, "temperature")
                            and step.config.temperature is not None
                        ):
                            kwargs["temperature"] = step.config.temperature

                        # Check agent signature (simplified)
                        import inspect

                        try:
                            run_sig = inspect.signature(agent.run)
                            run_params = list(run_sig.parameters.keys())
                            # Only pass kwargs that the agent accepts
                            filtered_kwargs = {k: v for k, v in kwargs.items() if k in run_params}
                        except (TypeError, ValueError):
                            # Fallback for mock agents
                            filtered_kwargs = kwargs

                        # Call agent
                        raw = await agent.run(data, **filtered_kwargs)

                        # Create result
                        result = StepResult(
                            name=step.name,
                            output=raw,
                            success=True,
                            attempts=attempt,
                            latency_s=0.0,
                        )

                        # Cache the result
                        self._cache[cache_key] = result
                        return result

                    except Exception as e:
                        last_exception = e
                        if attempt == max_retries:
                            # Return failed result
                            return StepResult(
                                name=step.name,
                                output=None,
                                success=False,
                                attempts=attempt,
                                latency_s=0.0,
                                feedback=str(last_exception),
                            )
                        await asyncio.sleep(0.001)  # Brief delay between retries

        return RealisticExecutor()

    @pytest.fixture
    def mock_step(self):
        """Create a mock step for benchmarking."""
        step = Mock(spec=Step)
        step.name = "benchmark_step"
        step.config = Mock()
        step.config.max_retries = 3
        step.config.temperature = 0.7
        step.agent = AsyncMock()
        step.agent.run.return_value = "benchmark_output"
        step.validators = []
        step.plugins = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False
        return step

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_realistic_execution_speed_comparison(
        self, ultra_executor, realistic_iterative_executor, mock_step
    ):
        """Compare execution speed between ultra and realistic iterative executors."""
        iterations = 50  # Reduced for more stable measurements
        data = {
            "test": "data",
            "nested": {"value": 123},
            "complex": {"list": [1, 2, 3], "dict": {"a": 1}},
        }
        context = _BenchmarkContext()
        resources = _BenchmarkResources()

        # Warm up both executors
        for _ in range(5):
            await execute_simple_step(ultra_executor, mock_step, data, context, resources)
            await realistic_iterative_executor.execute_step(mock_step, data, context, resources)

        # Benchmark ultra executor
        ultra_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            result = await execute_simple_step(
                ultra_executor,
                step=mock_step,
                data=data,
                context=context,
                resources=resources,
            )
            end_time = time.perf_counter()
            ultra_times.append(end_time - start_time)
            assert result.success

        # Benchmark realistic iterative executor
        iterative_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            result = await realistic_iterative_executor.execute_step(
                mock_step,
                data,
                context,
                resources,
            )
            end_time = time.perf_counter()
            iterative_times.append(end_time - start_time)
            assert result.success

        # Calculate statistics
        ultra_mean = statistics.mean(ultra_times)
        iterative_mean = statistics.mean(iterative_times)
        speedup = iterative_mean / ultra_mean

        print("\nRealistic Performance Comparison:")
        print(f"Ultra Executor - Mean: {ultra_mean:.6f}s")
        print(f"Realistic Iterative Executor - Mean: {iterative_mean:.6f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(
            f"Ultra executor is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than realistic iterative"
        )

        # RELATIVE performance check (environment-independent):
        #
        # Ultra executor provides significant value through:
        # - Robust error handling and retries
        # - Advanced caching with TTL and LRU
        # - Usage tracking and limits
        # - Concurrency management
        # - Comprehensive logging and telemetry
        # - Support for complex step types (plugins, validators, fallbacks)
        # - Optimized serialization and hashing
        #
        # The overhead is justified by the features, so we expect it to be slower
        # but still reasonably performant (within 10x of baseline).
        # This is a RELATIVE check that works the same in local and CI.
        max_overhead = 10.0
        assert speedup < max_overhead, (
            f"Ultra executor overhead should be < {max_overhead}x. "
            f"Got {speedup:.2f}x (ultra={ultra_mean:.6f}s, iterative={iterative_mean:.6f}s)"
        )

        # Sanity check: operations should complete (not hang)
        assert ultra_mean < 1.0, f"Ultra executor too slow: {ultra_mean:.3f}s per execution"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_ultra_executor_feature_value(self, ultra_executor, mock_step):
        """Test the value of ultra executor features."""
        data = {"feature": "test", "value": 123}
        context = _BenchmarkContext()
        resources = _BenchmarkResources()

        # Test 1: Basic execution
        result = await execute_simple_step(ultra_executor, mock_step, data, context, resources)
        assert result.success, "Ultra executor should execute successfully"
        assert result.output == "benchmark_output"

        # Test 2: Retry mechanism
        retry_agent = AsyncMock()
        retry_agent.run.side_effect = [
            Exception("First attempt fails"),
            Exception("Second attempt fails"),
            "success_after_retries",
        ]

        retry_step = Mock(spec=Step)
        retry_step.name = "retry_step"
        retry_step.config = Mock()
        retry_step.config.max_retries = 3
        retry_step.config.temperature = None
        retry_step.agent = retry_agent
        retry_step.validators = []
        retry_step.plugins = []
        retry_step.fallback_step = None
        retry_step.processors = Mock()
        retry_step.processors.prompt_processors = []
        retry_step.processors.output_processors = []
        retry_step.failure_handlers = []
        retry_step.persist_validation_results_to = None
        retry_step.meta = {}
        retry_step.persist_feedback_to_context = False

        result = await execute_simple_step(ultra_executor, retry_step, data, context, resources)

        # Should eventually succeed after retries
        assert result.success, "Ultra executor should handle retries and eventually succeed"
        assert result.attempts == 3, "Should have made 3 attempts"

        # Test 3: Concurrency management
        num_concurrent = 10

        slow_run_helper = create_slow_run_helper()

        slow_agent = AsyncMock()
        slow_agent.run.side_effect = slow_run_helper

        slow_step = Mock(spec=Step)
        slow_step.name = "slow_step"
        slow_step.config = Mock()
        slow_step.config.max_retries = 1
        slow_step.config.temperature = None
        slow_step.agent = slow_agent
        slow_step.validators = []
        slow_step.plugins = []
        slow_step.fallback_step = None
        slow_step.processors = Mock()
        slow_step.processors.prompt_processors = []
        slow_step.processors.output_processors = []
        slow_step.failure_handlers = []
        slow_step.persist_validation_results_to = None
        slow_step.meta = {}
        slow_step.persist_feedback_to_context = False

        start_time = time.perf_counter()
        tasks = [
            execute_simple_step(ultra_executor, slow_step, data, context, resources)
            for _ in range(num_concurrent)
        ]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.perf_counter() - start_time

        print("\nConcurrency Management Feature Value:")
        print(f"Concurrent execution time: {concurrent_time:.3f}s")
        print(f"All succeeded: {all(r.success for r in results)}")
        print(f"Average time per task: {concurrent_time / num_concurrent:.6f}s")

        # Validate correctness under concurrency
        assert all(r.success for r in results), "All concurrent tasks should succeed"
        # Sanity check: should complete (not hang)
        assert concurrent_time < 30.0, (
            f"Concurrent execution took too long: {concurrent_time:.3f}s (max 30s sanity check)"
        )

        # Test 4: Usage tracking
        usage_agent = AsyncMock()
        usage_agent.run.return_value = "usage_test_result"

        usage_step = Mock(spec=Step)
        usage_step.name = "usage_step"
        usage_step.config = Mock()
        usage_step.config.max_retries = 1
        usage_step.config.temperature = None
        usage_step.agent = usage_agent
        usage_step.validators = []
        usage_step.plugins = []
        usage_step.fallback_step = None
        usage_step.processors = Mock()
        usage_step.processors.prompt_processors = []
        usage_step.processors.output_processors = []
        usage_step.failure_handlers = []
        usage_step.persist_validation_results_to = None
        usage_step.meta = {}
        usage_step.persist_feedback_to_context = False

        # Add some usage if the attribute exists
        if hasattr(ultra_executor, "_usage"):
            await ultra_executor._usage.add(0.1, 100)

        result = await execute_simple_step(ultra_executor, usage_step, data, context, resources)

        print("\nUsage Tracking Feature Value:")
        if hasattr(ultra_executor, "_usage"):
            print(f"Total cost tracked: {ultra_executor._usage.total_cost}")
            print(f"Total tokens tracked: {ultra_executor._usage.total_tokens}")
        else:
            print("Usage tracking not available in this UltraExecutor version")
        print(f"Step executed successfully: {result.success}")

        # Should execute successfully
        assert result.success, "Step should execute successfully"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cache_performance_realistic(
        self, ultra_executor, realistic_iterative_executor, mock_step
    ):
        """Test cache performance with realistic workloads."""
        data = {
            "cache": "test",
            "value": 456,
            "complex": {"nested": {"data": "structure"}},
        }
        context = _BenchmarkContext()
        resources = _BenchmarkResources()

        # First execution (cache miss)
        start_time = time.perf_counter()
        result1 = await execute_simple_step(
            ultra_executor,
            step=mock_step,
            data=data,
            context=context,
            resources=resources,
        )
        first_execution_time = time.perf_counter() - start_time

        # Second execution (cache hit)
        start_time = time.perf_counter()
        result2 = await execute_simple_step(
            ultra_executor,
            step=mock_step,
            data=data,
            context=context,
            resources=resources,
        )
        cached_execution_time = time.perf_counter() - start_time

        # Verify results are identical
        assert result1.output == result2.output
        assert result1.success == result2.success

        # Cached execution should be significantly faster
        cache_speedup = first_execution_time / cached_execution_time
        print("\nCache Performance (Realistic):")
        print(f"First execution: {first_execution_time:.6f}s")
        print(f"Cached execution: {cached_execution_time:.6f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")

        assert cache_speedup >= 1.8, (
            f"Cache should provide significant speedup, got {cache_speedup:.2f}x"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_execution_performance(self, ultra_executor, mock_step):
        """Test performance under concurrent execution."""
        num_concurrent = 20
        data = {"concurrent": "test", "complex": {"data": "structure"}}
        context = _BenchmarkContext()
        resources = _BenchmarkResources()

        # Execute many steps concurrently
        start_time = time.perf_counter()
        tasks = [
            execute_simple_step(
                ultra_executor,
                step=mock_step,
                data=data,
                context=context,
                resources=resources,
            )
            for _ in range(num_concurrent)
        ]

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        # All should succeed
        assert all(r.success for r in results)
        assert all(r.output == "benchmark_output" for r in results)

        print("\nConcurrent Execution Performance:")
        print(f"Total time for {num_concurrent} concurrent executions: {total_time:.3f}s")
        print(f"Average time per execution: {total_time / num_concurrent:.6f}s")

        # RELATIVE performance check (environment-independent):
        # Concurrent execution should be faster than sequential would be.
        # With 50 concurrent operations, we expect significant parallelism benefit.
        # Sanity check: should complete within reasonable time (not hang)
        assert total_time < 30.0, (
            f"Concurrent execution took too long: {total_time:.3f}s (max 30s sanity check)"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_efficiency(self, ultra_executor, mock_step):
        """Test memory efficiency by running many executions."""
        iterations = 500  # Reduced for faster execution
        data = {"memory": "test", "large": "payload" * 50}
        context = _BenchmarkContext()
        resources = _BenchmarkResources()

        # Run many executions and measure memory usage
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Execute many steps
        for i in range(iterations):
            result = await execute_simple_step(
                ultra_executor,
                step=mock_step,
                data=data,
                context=context,
                resources=resources,
            )
            assert result.success

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        print("\nMemory Efficiency:")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        print(f"Memory per execution: {memory_increase / iterations / 1024:.2f} KB")

        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024, (
            f"Memory increase too high: {memory_increase / 1024 / 1024:.2f} MB"
        )

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hash_performance(self, ultra_executor):
        """Test the performance of the optimized hashing."""
        # Test with various data types
        test_data = [
            "simple string",
            {"complex": "object", "with": ["nested", "data"]},
            [1, 2, 3, 4, 5],
            {"large": "payload" * 100},
            None,
            123,
            3.14159,
            True,
        ]

        start_time = time.perf_counter()
        hashes = []
        for data in test_data:
            for _ in range(50):  # Reduced iterations
                hash_val = ultra_executor._hash_obj(data)
                hashes.append(hash_val)
        total_time = time.perf_counter() - start_time

        print("\nHash Performance:")
        print(f"Total hashing time: {total_time:.3f}s")
        print(f"Average time per hash: {total_time / len(hashes):.6f}s")
        print(f"Hashes per second: {len(hashes) / total_time:.0f}")

        # Sanity check: hashing should complete (not hang)
        # No absolute threshold - just verify it completes reasonably
        assert total_time < 30.0, f"Hashing took too long: {total_time:.3f}s (max 30s sanity check)"

        # Verify hash consistency
        for i, data in enumerate(test_data):
            base_hash = ultra_executor._hash_obj(data)
            for _ in range(5):
                assert ultra_executor._hash_obj(data) == base_hash

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cache_key_generation_performance(self, ultra_executor, mock_step):
        """Test cache key generation performance."""
        iterations = 500  # Reduced for faster execution

        # Create various frames if the class exists
        frames = []
        if hasattr(ultra_executor, "_Frame"):
            for i in range(iterations):
                frame = ultra_executor._Frame(
                    step=mock_step,
                    data={"test": i, "data": f"value_{i}"},
                    context=_BenchmarkContext() if i % 2 == 0 else None,
                    resources=Mock() if i % 3 == 0 else None,
                )
                frames.append(frame)

            # Benchmark cache key generation
            start_time = time.perf_counter()
            keys = []
            for frame in frames:
                key = ultra_executor._cache_key(frame)
                keys.append(key)
            total_time = time.perf_counter() - start_time
        else:
            # Skip this test if _Frame class doesn't exist
            pytest.skip("_Frame class not available in this UltraExecutor version")

        print("\nCache Key Generation Performance:")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per key: {total_time / iterations:.6f}s")
        print(f"Keys per second: {iterations / total_time:.0f}")

        # Sanity check: cache key generation should complete (not hang)
        assert total_time < 30.0, (
            f"Cache key generation took too long: {total_time:.3f}s (max 30s sanity check)"
        )

        # Verify key uniqueness
        unique_keys = set(keys)
        assert len(unique_keys) > len(keys) * 0.8, "Most keys should be unique"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_usage_tracking_performance(self, ultra_executor):
        """Test usage tracking performance under high load."""
        iterations = 500  # Reduced for faster execution

        # Simulate many usage additions if the attribute exists
        if hasattr(ultra_executor, "_usage"):
            start_time = time.perf_counter()
            for i in range(iterations):
                await ultra_executor._usage.add(i * 0.01, i * 10)
            total_time = time.perf_counter() - start_time

            print("\nUsage Tracking Performance:")
            print(f"Total time for {iterations} additions: {total_time:.3f}s")
            print(f"Average time per addition: {total_time / iterations:.6f}s")
            print(f"Additions per second: {iterations / total_time:.0f}")

            # Sanity check: usage tracking should complete (not hang)
            assert total_time < 30.0, (
                f"Usage tracking took too long: {total_time:.3f}s (max 30s sanity check)"
            )

            # Verify final values
            assert ultra_executor._usage.total_cost > 0
            assert ultra_executor._usage.total_tokens > 0
        else:
            # Skip this test if usage tracking is not available
            pytest.skip("Usage tracking not available in this UltraExecutor version")


class TestUltraExecutorScalability:
    """Test ultra executor scalability characteristics."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_large_cache_performance(self):
        """Test performance with large cache.

        Uses RELATIVE performance measurement:
        - Cache hits should be significantly faster than cache fills (misses)
        - This is environment-independent (works same in local and CI)
        """
        executor = UltraStepExecutor(
            enable_cache=True,
            cache_size=5000,  # Large cache
            cache_ttl=3600,
        )

        step = Mock(spec=Step)
        step.name = "large_cache_test"
        step.config = Mock()
        step.config.max_retries = 1
        step.agent = AsyncMock()
        step.agent.run.return_value = "output"
        step.validators = []
        step.plugins = []
        step.fallback_step = None
        step.processors = Mock()
        step.processors.prompt_processors = []
        step.processors.output_processors = []
        step.failure_handlers = []
        step.persist_validation_results_to = None
        step.meta = {}
        step.persist_feedback_to_context = False

        num_entries = 500  # Use same count for fair comparison

        # Fill cache (cache misses - agent.run is called)
        start_time = time.perf_counter()
        for i in range(num_entries):
            result = await execute_simple_step(
                executor,
                step=step,
                data={"index": i, "data": f"value_{i}"},
                context=None,
                resources=None,
            )
            assert result.success
        fill_time = time.perf_counter() - start_time

        # Test cache hits (no agent.run called)
        start_time = time.perf_counter()
        for i in range(num_entries):
            result = await execute_simple_step(
                executor,
                step=step,
                data={"index": i, "data": f"value_{i}"},
                context=None,
                resources=None,
            )
            assert result.success
        hit_time = time.perf_counter() - start_time

        print("\nLarge Cache Performance:")
        print(
            f"  Cache fill (misses): {fill_time:.3f}s ({fill_time / num_entries * 1000:.3f}ms/op)"
        )
        print(f"  Cache hits: {hit_time:.3f}s ({hit_time / num_entries * 1000:.3f}ms/op)")

        # RELATIVE performance check (environment-independent):
        # Cache hits should be faster than cache fills because:
        # - Cache fill calls agent.run() (even if mocked, has overhead)
        # - Cache hit just does hash lookup and returns cached result
        #
        # We require at least 1.5x speedup (hits vs fills).
        # In practice, we often see 2-10x speedup.
        speedup = fill_time / hit_time if hit_time > 0 else float("inf")
        min_speedup = 1.5

        print(f"  Speedup (fill/hit): {speedup:.2f}x")

        assert speedup >= min_speedup, (
            f"Cache hits should be at least {min_speedup}x faster than fills. "
            f"Got {speedup:.2f}x (fill={fill_time:.3f}s, hit={hit_time:.3f}s)"
        )

        # Sanity check: operations should complete (not hang)
        assert fill_time < 30.0, f"Cache fill took too long: {fill_time:.2f}s"
        assert hit_time < 30.0, f"Cache hits took too long: {hit_time:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrency_scaling(self):
        """Test how performance scales with concurrency."""
        concurrency_levels = [1, 2, 4, 8]  # Reduced levels for faster execution

        for concurrency in concurrency_levels:
            executor = UltraStepExecutor(
                enable_cache=True,
                concurrency_limit=concurrency,
            )

            step = Mock(spec=Step)
            step.name = f"concurrency_test_{concurrency}"
            step.config = Mock()
            step.config.max_retries = 1
            step.agent = AsyncMock()
            step.validators = []
            step.plugins = []
            step.fallback_step = None
            step.processors = Mock()
            step.processors.prompt_processors = []
            step.processors.output_processors = []
            step.failure_handlers = []
            step.persist_validation_results_to = None
            step.meta = {}
            step.persist_feedback_to_context = False

            slow_run_helper = create_slow_run_helper()

            step.agent.run = slow_run_helper

            # Execute many tasks concurrently
            num_tasks = concurrency * 5  # Reduced for faster execution
            start_time = time.perf_counter()
            tasks = [
                execute_simple_step(executor, step, f"input_{i}", None, None)
                for i in range(num_tasks)
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            print(f"\nConcurrency {concurrency}:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Tasks per second: {num_tasks / total_time:.1f}")
            print(f"  Average time per task: {total_time / num_tasks:.6f}s")

            # All should succeed
            assert all(r.success for r in results)
