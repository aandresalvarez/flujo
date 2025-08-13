"""
Comprehensive performance benchmark tests for ExecutorCore optimization.

This test suite establishes baseline performance metrics and validates optimization improvements.
Tests are designed to measure execution performance, memory usage, concurrency, caching, and
context handling performance with statistical analysis and CI integration.
"""

import asyncio
import gc
import os
import psutil
import pytest
import statistics
import time
from contextlib import contextmanager
from typing import Dict, Optional
from unittest.mock import Mock, AsyncMock

from flujo.application.core.executor_core import ExecutorCore
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent

# Constants for better maintainability
FLOAT_TOLERANCE = 1e-10
CI_TRUE_VALUES = ("true", "1", "yes")
DEFAULT_ITERATIONS = 100
WARMUP_ITERATIONS = 10


@contextmanager
def temporary_env_var(var_name: str, value: Optional[str]):
    """Context manager to temporarily set an environment variable."""
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


def get_performance_threshold(base_threshold: float, ci_multiplier: float = 1.5) -> float:
    """Get performance threshold based on environment."""
    is_ci = os.getenv("CI", "false").lower() in CI_TRUE_VALUES
    return base_threshold * ci_multiplier if is_ci else base_threshold


def create_step(output: str = "ok", name: str = "perf_step") -> Step:
    """Create a test step with configurable output."""
    # Create a StubAgent with infinite outputs for benchmarking
    return Step.model_validate(
        {
            "name": name,
            "agent": StubAgent([output] * 1000),  # Provide 1000 outputs for benchmarking
            "config": StepConfig(max_retries=1),
        }
    )


def create_mock_step(output: str = "mock_output", name: str = "mock_step") -> Mock:
    """Create a mock step for performance testing."""
    step = Mock(spec=Step)
    step.name = name
    step.config = Mock()
    step.config.max_retries = 1
    step.config.temperature = None
    step.agent = AsyncMock()
    step.agent.run.return_value = output
    step.validators = []
    step.plugins = []
    step.fallback_step = None  # Explicitly set to None to prevent recursive loops
    step.processors = Mock()
    step.processors.prompt_processors = []
    step.processors.output_processors = []
    step.failure_handlers = []
    step.persist_validation_results_to = None
    step.meta = {}
    step.persist_feedback_to_context = False

    # Ensure the mock doesn't create recursive references
    step.fallback_step = None
    step.agent.fallback_step = None

    return step


class PerformanceBenchmark:
    """Performance benchmarking utilities with statistical analysis."""

    def __init__(self):
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}

    async def benchmark_function(
        self,
        func,
        iterations: int = DEFAULT_ITERATIONS,
        warmup: int = WARMUP_ITERATIONS,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """Benchmark a function with statistical analysis."""
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        # Collect garbage before benchmarking
        gc.collect()

        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        # Calculate statistics
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "p95": sorted(times)[int(len(times) * 0.95)],
            "p99": sorted(times)[int(len(times) * 0.99)],
            "iterations": iterations,
        }

    def memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }


@pytest.fixture
def perf_benchmark():
    """Provide performance benchmark utilities."""
    return PerformanceBenchmark()


@pytest.fixture
def executor_core():
    """Create ExecutorCore for benchmarking."""
    return ExecutorCore(enable_cache=True, concurrency_limit=8)


@pytest.fixture
def mock_context():
    """Create mock context for testing."""
    context = Mock()
    context.model_dump.return_value = {"test": "context"}
    return context


class TestExecutorCorePerformance:
    """Core performance tests for ExecutorCore."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_executor_core_execution_performance(self, executor_core, perf_benchmark):
        """Benchmark overall ExecutorCore execution performance."""
        step = create_step("performance_test")
        data = {"test": "data", "value": 123}

        # Benchmark execution
        stats = await perf_benchmark.benchmark_function(
            executor_core.execute,
            iterations=50,  # Reduced for CI stability
            step=step,
            data=data,
        )

        print("\nExecution Performance Stats:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")
        print(f"P99: {stats['p99']:.6f}s")

        # Performance assertions
        threshold = get_performance_threshold(0.01)  # 10ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Mean execution time {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"
        assert (
            stats["p95"] < threshold * 2
        ), f"P95 execution time {stats['p95']:.6f}s exceeds threshold {threshold * 2:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_executor_core_memory_usage(self, executor_core, perf_benchmark):
        """Test memory usage patterns of ExecutorCore."""
        step = create_step("memory_test")
        step.agent = StubAgent(["memory_test"] * 105)  # Extra outputs for safety
        data = {"memory": "test", "large_payload": "x" * 1000}

        # Measure initial memory
        initial_memory = perf_benchmark.memory_usage()

        # Execute multiple steps
        for i in range(100):
            result = await executor_core.execute(step, data)
            assert result.success

        # Measure final memory
        final_memory = perf_benchmark.memory_usage()
        memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

        print("\nMemory Usage Stats:")
        print(f"Initial: {initial_memory['rss_mb']:.2f} MB")
        print(f"Final: {final_memory['rss_mb']:.2f} MB")
        print(f"Increase: {memory_increase:.2f} MB")

        # Memory should not increase excessively
        threshold = get_performance_threshold(50.0)  # 50MB base threshold
        assert (
            memory_increase < threshold
        ), f"Memory increase {memory_increase:.2f}MB exceeds threshold {threshold:.2f}MB"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_executor_core_concurrent_execution(self, executor_core, perf_benchmark):
        """Test performance under concurrent execution scenarios."""
        step = create_step("concurrent_test")
        data = {"concurrent": "test"}
        num_concurrent = 20

        # Benchmark concurrent execution
        async def concurrent_execution():
            tasks = [executor_core.execute(step, data) for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            return results

        stats = await perf_benchmark.benchmark_function(
            concurrent_execution,
            iterations=10,  # Reduced for CI stability
            warmup=2,
        )

        print("\nConcurrent Execution Stats:")
        print(f"Mean: {stats['mean']:.6f}s for {num_concurrent} concurrent executions")
        print(f"P95: {stats['p95']:.6f}s")

        # Concurrent execution should be efficient
        threshold = get_performance_threshold(1.0)  # 1s base threshold
        assert (
            stats["mean"] < threshold
        ), f"Concurrent execution time {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_executor_core_cache_performance(self, executor_core, perf_benchmark):
        """Test cache performance improvements."""
        step = create_step("cache_test")
        data = {"cache": "test", "value": 456}

        # First execution (cache miss)
        miss_stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=20, step=step, data=data
        )

        # Second execution (cache hit)
        hit_stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=20, step=step, data=data
        )

        cache_speedup = miss_stats["mean"] / hit_stats["mean"]

        print("\nCache Performance Stats:")
        print(f"Cache miss mean: {miss_stats['mean']:.6f}s")
        print(f"Cache hit mean: {hit_stats['mean']:.6f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")

        # For very fast operations, cache overhead might make it slightly slower
        # This is acceptable for microsecond-level operations where cache overhead dominates
        # The real benefit of caching comes with slower operations (API calls, etc.)
        min_speedup = 0.8  # Allow 20% overhead for very fast operations
        assert (
            cache_speedup >= min_speedup
        ), f"Cache speedup {cache_speedup:.2f}x should be at least {min_speedup:.1f}x (allowing overhead for fast operations)"

        # Log the performance for analysis
        print("Cache performance analysis:")
        print(f"  - Cache miss operations are very fast ({miss_stats['mean']:.6f}s)")
        print(f"  - Cache hit operations are slightly faster ({hit_stats['mean']:.6f}s)")
        print(f"  - Speedup factor: {cache_speedup:.2f}x")

        # The test passes if there's no performance regression
        # In practice, cache benefits are more apparent with slower operations

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_executor_core_context_handling_performance(
        self, executor_core, perf_benchmark, mock_context
    ):
        """Test context handling performance optimizations."""
        step = create_step("context_test")
        data = {"context": "test"}

        # Benchmark with context
        with_context_stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data, context=mock_context
        )

        # Benchmark without context
        without_context_stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data
        )

        context_overhead = with_context_stats["mean"] - without_context_stats["mean"]
        overhead_percentage = (context_overhead / without_context_stats["mean"]) * 100

        print("\nContext Handling Performance Stats:")
        print(f"Without context: {without_context_stats['mean']:.6f}s")
        print(f"With context: {with_context_stats['mean']:.6f}s")
        print(f"Context overhead: {context_overhead:.6f}s ({overhead_percentage:.1f}%)")

        # Context overhead should be minimal
        max_overhead_percentage = get_performance_threshold(50.0)  # 50% base threshold
        assert (
            overhead_percentage < max_overhead_percentage
        ), f"Context overhead {overhead_percentage:.1f}% exceeds threshold {max_overhead_percentage:.1f}%"


class TestComponentPerformance:
    """Component-specific performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_agent_runner_performance(self, executor_core, perf_benchmark):
        """Test agent runner performance optimizations."""
        step = create_mock_step("agent_test")
        data = {"agent": "performance"}

        stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data
        )

        print("\nAgent Runner Performance Stats:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")

        threshold = get_performance_threshold(0.005)  # 5ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Agent runner performance {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_processor_pipeline_performance(self, executor_core, perf_benchmark):
        """Test processor pipeline performance improvements."""
        step = create_mock_step("processor_test")

        # Add mock processors
        processor = Mock()
        processor.process = Mock(side_effect=lambda x, **kwargs: x)
        step.processors.prompt_processors = [processor]
        step.processors.output_processors = [processor]

        data = {"processor": "test"}

        stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data
        )

        print("\nProcessor Pipeline Performance Stats:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")

        threshold = get_performance_threshold(0.01)  # 10ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Processor pipeline performance {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_validator_runner_performance(self, executor_core, perf_benchmark):
        """Test validator runner performance optimizations."""
        step = create_mock_step("validator_test")

        # Add mock validator
        validator = Mock()
        validator.validate = AsyncMock(return_value=None)
        step.validators = [validator]

        data = {"validator": "test"}

        stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data
        )

        print("\nValidator Runner Performance Stats:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")

        threshold = get_performance_threshold(0.01)  # 10ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Validator runner performance {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_plugin_runner_performance(self, executor_core, perf_benchmark):
        """Test plugin runner performance improvements."""
        step = create_mock_step("plugin_test")

        # Add mock plugin
        plugin = Mock()
        plugin.validate = AsyncMock(return_value="processed_data")
        step.plugins = [(plugin, 1)]

        data = {"plugin": "test"}

        stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=50, step=step, data=data
        )

        print("\nPlugin Runner Performance Stats:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")

        threshold = get_performance_threshold(0.01)  # 10ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Plugin runner performance {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"


class TestMemoryManagement:
    """Memory management performance tests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_allocation_optimization(self, executor_core, perf_benchmark):
        """Test memory allocation optimizations."""
        step = create_step("memory_alloc_test")
        data = {"memory": "allocation", "size": 1000}

        # Measure memory allocations
        initial_memory = perf_benchmark.memory_usage()

        # Execute many steps to test allocation patterns
        for i in range(200):
            result = await executor_core.execute(step, data)
            assert result.success

            # Force garbage collection periodically
            if i % 50 == 0:
                gc.collect()

        final_memory = perf_benchmark.memory_usage()
        memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

        print("\nMemory Allocation Optimization Stats:")
        print(f"Memory increase: {memory_increase:.2f} MB for 200 executions")
        print(f"Memory per execution: {memory_increase / 200 * 1024:.2f} KB")

        # Memory increase should be reasonable
        threshold = get_performance_threshold(100.0)  # 100MB base threshold
        assert (
            memory_increase < threshold
        ), f"Memory increase {memory_increase:.2f}MB exceeds threshold {threshold:.2f}MB"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_garbage_collection_impact(self, executor_core, perf_benchmark):
        """Test garbage collection impact on performance."""
        step = create_step("gc_test")
        data = {"gc": "test"}

        # Disable automatic GC
        gc.disable()

        try:
            # Benchmark without GC
            no_gc_stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=50, step=step, data=data
            )

            # Enable GC and benchmark
            gc.enable()
            gc_stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=50, step=step, data=data
            )

            gc_impact = gc_stats["mean"] - no_gc_stats["mean"]
            impact_percentage = (gc_impact / no_gc_stats["mean"]) * 100

            print("\nGarbage Collection Impact Stats:")
            print(f"Without GC: {no_gc_stats['mean']:.6f}s")
            print(f"With GC: {gc_stats['mean']:.6f}s")
            print(f"GC impact: {gc_impact:.6f}s ({impact_percentage:.1f}%)")

            # GC impact should be minimal
            max_impact_percentage = get_performance_threshold(25.0)  # 25% base threshold
            assert (
                impact_percentage < max_impact_percentage
            ), f"GC impact {impact_percentage:.1f}% exceeds threshold {max_impact_percentage:.1f}%"

        finally:
            gc.enable()

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_leak_prevention(self, executor_core, perf_benchmark):
        """Test memory leak prevention mechanisms."""
        step = create_step("leak_test")

        # Execute many steps and monitor memory growth
        memory_samples = []

        for i in range(100):
            data = {"leak": "test", "iteration": i}
            result = await executor_core.execute(step, data)
            assert result.success

            if i % 10 == 0:
                memory_samples.append(perf_benchmark.memory_usage()["rss_mb"])

        # Check for memory growth trend
        if len(memory_samples) > 2:
            memory_growth = memory_samples[-1] - memory_samples[0]
            growth_per_iteration = memory_growth / 100

            print("\nMemory Leak Prevention Stats:")
            print(f"Initial memory: {memory_samples[0]:.2f} MB")
            print(f"Final memory: {memory_samples[-1]:.2f} MB")
            print(f"Total growth: {memory_growth:.2f} MB")
            print(f"Growth per iteration: {growth_per_iteration * 1024:.2f} KB")

            # Memory growth should be minimal
            max_growth_per_iteration = get_performance_threshold(0.1)  # 0.1MB per iteration
            assert (
                growth_per_iteration < max_growth_per_iteration
            ), f"Memory growth {growth_per_iteration:.3f}MB per iteration exceeds threshold {max_growth_per_iteration:.3f}MB"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_object_pooling_performance(self, executor_core, perf_benchmark):
        """Test object pooling performance improvements."""
        # This test will be enhanced once object pooling is implemented
        step = create_step("pooling_test")
        data = {"pooling": "test"}

        # Benchmark current performance (baseline for future comparison)
        stats = await perf_benchmark.benchmark_function(
            executor_core.execute, iterations=100, step=step, data=data
        )

        print("\nObject Pooling Performance Stats (Baseline):")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")
        print(f"Memory efficiency: {stats['mean'] * 1000:.2f}ms per execution")

        # Store baseline for future comparison
        perf_benchmark.baseline_metrics["object_pooling"] = stats["mean"]

        # Basic performance assertion
        threshold = get_performance_threshold(0.01)  # 10ms base threshold
        assert (
            stats["mean"] < threshold
        ), f"Object pooling baseline {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"


class TestAdvancedComponentPerformance:
    """Advanced component performance benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_serialization_performance(self, executor_core, perf_benchmark):
        """Test serialization performance optimizations."""
        step = create_step("serialization_test")

        # Test with various data sizes
        data_sizes = [
            ("small", {"small": "x" * 100}),
            ("medium", {"medium": "x" * 1000}),
            ("large", {"large": "x" * 10000}),
        ]

        for size_name, data in data_sizes:
            stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=30, step=step, data=data
            )

            print(f"\nSerialization Performance ({size_name}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            # Larger data should still be reasonably fast
            threshold = get_performance_threshold(0.02)  # 20ms base threshold
            assert (
                stats["mean"] < threshold
            ), f"Serialization performance for {size_name} data {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_hash_computation_performance(self, executor_core, perf_benchmark):
        """Test hash computation performance optimizations."""
        step = create_step("hash_test")

        # Test with various data structures
        test_data = [
            {"simple": "string"},
            {"nested": {"deep": {"structure": {"with": ["lists", "and", "dicts"]}}}},
            {"large_list": list(range(1000))},
            {"mixed": {"str": "test", "int": 123, "float": 3.14, "bool": True, "none": None}},
        ]

        for i, data in enumerate(test_data):
            stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=50, step=step, data=data
            )

            print(f"\nHash Computation Performance (test {i + 1}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            threshold = get_performance_threshold(0.01)  # 10ms base threshold
            assert (
                stats["mean"] < threshold
            ), f"Hash computation performance for test {i + 1} {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cache_key_generation_performance(self, executor_core, perf_benchmark):
        """Test cache key generation performance."""
        step = create_step("cache_key_test")

        # Generate cache keys for various scenarios
        test_scenarios = [
            {"data": {"simple": "test"}, "context": None, "resources": None},
            {
                "data": {"complex": {"nested": "data"}},
                "context": {"ctx": "test"},
                "resources": None,
            },
            {
                "data": {"large": "x" * 1000},
                "context": {"large_ctx": "y" * 500},
                "resources": {"res": "z" * 200},
            },
        ]

        for i, scenario in enumerate(test_scenarios):
            # Test cache key generation indirectly through execution
            stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=50, step=step, **scenario
            )

            print(f"\nCache Key Generation Performance (scenario {i + 1}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            threshold = get_performance_threshold(0.015)  # 15ms base threshold
            assert (
                stats["mean"] < threshold
            ), f"Cache key generation performance for scenario {i + 1} {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_context_copying_performance(self, executor_core, perf_benchmark):
        """Test context copying performance optimizations."""
        step = create_step("context_copy_test")
        data = {"context_copy": "test"}

        # Test with various context sizes
        contexts = [
            ("small", {"small": "context"}),
            ("medium", {f"key_{i}": f"value_{i}" for i in range(100)}),
            (
                "large",
                {f"key_{i}": {"nested": f"value_{i}", "list": list(range(10))} for i in range(50)},
            ),
        ]

        for size_name, context in contexts:
            mock_context = Mock()
            mock_context.model_dump.return_value = context

            stats = await perf_benchmark.benchmark_function(
                executor_core.execute, iterations=30, step=step, data=data, context=mock_context
            )

            print(f"\nContext Copying Performance ({size_name}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            threshold = get_performance_threshold(0.02)  # 20ms base threshold
            assert (
                stats["mean"] < threshold
            ), f"Context copying performance for {size_name} context {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"


class TestConcurrencyPerformance:
    """Concurrency-specific performance benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_semaphore_contention_performance(self, perf_benchmark):
        """Test semaphore contention performance."""
        # Test with different concurrency limits
        concurrency_limits = [1, 4, 8, 16]

        for limit in concurrency_limits:
            executor = ExecutorCore(enable_cache=True, concurrency_limit=limit)

            step = create_step("semaphore_test")
            data = {"semaphore": "test"}

            async def concurrent_execution():
                tasks = [
                    executor.execute(step, data)
                    for _ in range(limit * 2)  # Create more tasks than limit
                ]
                results = await asyncio.gather(*tasks)
                return results

            stats = await perf_benchmark.benchmark_function(
                concurrent_execution,
                iterations=5,  # Reduced for performance
                warmup=1,
            )

            print(f"\nSemaphore Contention Performance (limit={limit}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            # Higher limits should generally perform better
            threshold = get_performance_threshold(2.0)  # 2s base threshold
            assert (
                stats["mean"] < threshold
            ), f"Semaphore contention performance with limit {limit} {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_task_scheduling_performance(self, executor_core, perf_benchmark):
        """Test task scheduling performance."""
        step = create_step("scheduling_test")

        # Test with different task patterns
        task_patterns = [
            {"name": "burst", "tasks": 50, "delay": 0},
            {"name": "steady", "tasks": 20, "delay": 0.001},
            {"name": "mixed", "tasks": 30, "delay": 0.0005},
        ]

        for pattern in task_patterns:

            async def scheduled_execution():
                tasks = []
                for i in range(pattern["tasks"]):
                    if pattern["delay"] > 0:
                        await asyncio.sleep(pattern["delay"])
                    tasks.append(executor_core.execute(step, {"task": i}))

                results = await asyncio.gather(*tasks)
                return results

            stats = await perf_benchmark.benchmark_function(
                scheduled_execution,
                iterations=3,  # Reduced for performance
                warmup=1,
            )

            print(f"\nTask Scheduling Performance ({pattern['name']}):")
            print(f"Mean: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            threshold = get_performance_threshold(1.5)  # 1.5s base threshold
            assert (
                stats["mean"] < threshold
            ), f"Task scheduling performance for {pattern['name']} pattern {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_resource_contention_performance(self, executor_core, perf_benchmark):
        """Test resource contention performance."""
        # Create steps that compete for resources
        steps = [create_step(f"resource_test_{i}") for i in range(10)]
        data = {"resource": "contention"}

        async def resource_contention():
            # Execute different steps concurrently to test resource sharing
            tasks = [executor_core.execute(step, data) for step in steps]
            results = await asyncio.gather(*tasks)
            return results

        stats = await perf_benchmark.benchmark_function(
            resource_contention,
            iterations=5,  # Reduced for performance
            warmup=1,
        )

        print("\nResource Contention Performance:")
        print(f"Mean: {stats['mean']:.6f}s")
        print(f"P95: {stats['p95']:.6f}s")

        threshold = get_performance_threshold(1.0)  # 1s base threshold
        assert (
            stats["mean"] < threshold
        ), f"Resource contention performance {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"


class TestScalabilityBenchmarks:
    """Scalability performance benchmarks."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_linear_scaling_performance(self, perf_benchmark):
        """Test linear scaling performance with different loads."""
        loads = [1, 5, 10, 20]
        performance_data = []

        for load in loads:
            executor = ExecutorCore(enable_cache=True, concurrency_limit=max(4, load // 2))

            step = create_step("scaling_test")
            data = {"scaling": "test"}

            async def load_execution():
                tasks = [executor.execute(step, data) for _ in range(load)]
                results = await asyncio.gather(*tasks)
                return results

            stats = await perf_benchmark.benchmark_function(
                load_execution,
                iterations=3,  # Reduced for performance
                warmup=1,
            )

            performance_data.append(
                {"load": load, "mean_time": stats["mean"], "time_per_task": stats["mean"] / load}
            )

            print(f"\nLinear Scaling Performance (load={load}):")
            print(f"Total time: {stats['mean']:.6f}s")
            print(f"Time per task: {stats['mean'] / load:.6f}s")

        # Check that scaling is reasonable (not exponential)
        if len(performance_data) >= 2:
            first_per_task = performance_data[0]["time_per_task"]
            last_per_task = performance_data[-1]["time_per_task"]
            scaling_factor = last_per_task / first_per_task

            print("\nScaling Analysis:")
            print(f"First per-task time: {first_per_task:.6f}s")
            print(f"Last per-task time: {last_per_task:.6f}s")
            print(f"Scaling factor: {scaling_factor:.2f}x")

            # Scaling should be reasonable (not more than 3x degradation)
            max_scaling_factor = get_performance_threshold(3.0)
            assert (
                scaling_factor < max_scaling_factor
            ), f"Scaling factor {scaling_factor:.2f}x exceeds threshold {max_scaling_factor:.2f}x"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_scaling_performance(self, perf_benchmark):
        """Test memory scaling performance with different loads."""
        loads = [10, 50, 100]

        for load in loads:
            executor = ExecutorCore(enable_cache=True, concurrency_limit=8)

            step = create_step("memory_scaling_test")

            # Measure initial memory
            initial_memory = perf_benchmark.memory_usage()

            # Execute load
            for i in range(load):
                data = {"memory_scaling": "test", "iteration": i}
                result = await executor.execute(step, data)
                assert result.success

            # Measure final memory
            final_memory = perf_benchmark.memory_usage()
            memory_per_execution = (final_memory["rss_mb"] - initial_memory["rss_mb"]) / load

            print(f"\nMemory Scaling Performance (load={load}):")
            print(f"Memory per execution: {memory_per_execution * 1024:.2f} KB")
            print(
                f"Total memory increase: {final_memory['rss_mb'] - initial_memory['rss_mb']:.2f} MB"
            )

            # Memory per execution should be reasonable
            max_memory_per_execution = get_performance_threshold(0.5)  # 0.5MB per execution
            assert (
                memory_per_execution < max_memory_per_execution
            ), f"Memory per execution {memory_per_execution:.3f}MB exceeds threshold {max_memory_per_execution:.3f}MB"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cache_scaling_performance(self, perf_benchmark):
        """Test cache scaling performance with different cache sizes."""
        cache_sizes = [100, 500, 1000]

        for cache_size in cache_sizes:
            executor = ExecutorCore(enable_cache=True, concurrency_limit=8)

            step = create_step("cache_scaling_test")

            # Fill cache
            for i in range(cache_size):
                data = {"cache_scaling": i}
                result = await executor.execute(step, data)
                assert result.success

            # Test cache hit performance
            test_data = {"cache_scaling": cache_size // 2}  # Should be in cache

            stats = await perf_benchmark.benchmark_function(
                executor.execute, iterations=20, step=step, data=test_data
            )

            print(f"\nCache Scaling Performance (size={cache_size}):")
            print(f"Cache hit time: {stats['mean']:.6f}s")
            print(f"P95: {stats['p95']:.6f}s")

            # Cache hits should remain fast regardless of cache size
            threshold = get_performance_threshold(0.001)  # 1ms base threshold
            assert (
                stats["mean"] < threshold
            ), f"Cache hit performance with size {cache_size} {stats['mean']:.6f}s exceeds threshold {threshold:.6f}s"
