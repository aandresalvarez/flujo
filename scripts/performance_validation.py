#!/usr/bin/env python3
"""
Performance validation script for ExecutorCore optimization.

This script runs comprehensive performance benchmarks, collects metrics,
compares against baseline targets, and identifies optimization opportunities.
"""

import asyncio
import json
import psutil
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, List
from unittest.mock import Mock

# Import the optimized components
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
from flujo.domain.dsl.step import Step, StepConfig
from flujo.testing.utils import StubAgent


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    test_name: str
    mean_time: float
    median_time: float
    p95_time: float
    p99_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int
    memory_usage_mb: float
    success_rate: float
    throughput_ops_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ValidationResults:
    """Complete validation results."""

    baseline_metrics: dict[str, PerformanceMetrics]
    optimized_metrics: dict[str, PerformanceMetrics]
    improvements: dict[str, float]
    bottlenecks: List[str]
    recommendations: List[str]
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "baseline_metrics": {k: v.to_dict() for k, v in self.baseline_metrics.items()},
            "optimized_metrics": {k: v.to_dict() for k, v in self.optimized_metrics.items()},
            "improvements": self.improvements,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "overall_score": self.overall_score,
        }


class PerformanceValidator:
    """Comprehensive performance validation framework."""

    def __init__(self):
        self.baseline_executor = ExecutorCore(enable_cache=True, concurrency_limit=8)
        self.optimized_executor = self._create_optimized_executor()
        self.results = ValidationResults(
            baseline_metrics={},
            optimized_metrics={},
            improvements={},
            bottlenecks=[],
            recommendations=[],
            overall_score=0.0,
        )

    def _create_optimized_executor(self) -> ExecutorCore:
        """Create optimized executor with all optimizations enabled."""
        config = OptimizationConfig(
            enable_object_pool=True,
            enable_context_optimization=True,
            enable_memory_optimization=True,
            enable_optimized_telemetry=True,
            enable_performance_monitoring=True,
            enable_optimized_error_handling=True,
            enable_circuit_breaker=True,
            enable_automatic_optimization=True,
            max_concurrent_executions=50,
        )
        return ExecutorCore(optimization_config=config)

    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        yield
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        self._current_memory_usage = final_memory - initial_memory

    async def benchmark_function(
        self, func, iterations: int = 100, warmup: int = 10, *args, **kwargs
    ) -> PerformanceMetrics:
        """Benchmark a function with comprehensive metrics."""
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        # Benchmark with memory monitoring
        times = []
        successful_runs = 0

        with self.memory_monitor():
            for _ in range(iterations):
                start_time = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        func(*args, **kwargs)
                    successful_runs += 1
                except Exception as e:
                    print(f"Benchmark error: {e}")
                    continue
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        if not times:
            raise RuntimeError("No successful benchmark runs")

        # Calculate statistics
        mean_time = statistics.mean(times)
        return PerformanceMetrics(
            test_name="",  # Will be set by caller
            mean_time=mean_time,
            median_time=statistics.median(times),
            p95_time=sorted(times)[int(len(times) * 0.95)],
            p99_time=sorted(times)[int(len(times) * 0.99)],
            std_dev=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time=min(times),
            max_time=max(times),
            iterations=len(times),
            memory_usage_mb=getattr(self, "_current_memory_usage", 0.0),
            success_rate=successful_runs / iterations,
            throughput_ops_per_sec=1.0 / mean_time if mean_time > 0 else 0.0,
        )

    def create_test_step(self, name: str = "test_step", outputs: int = 200) -> Step:
        """Create a test step for benchmarking."""
        return Step.model_validate(
            {
                "name": name,
                "agent": StubAgent([f"output_{i}" for i in range(outputs)]),
                "config": StepConfig(max_retries=1),
            }
        )

    async def test_execution_performance(self) -> None:
        """Test basic execution performance."""
        step = self.create_test_step("execution_perf")
        data = {"test": "execution_performance"}

        # Baseline
        baseline_metrics = await self.benchmark_function(
            self.baseline_executor.execute, iterations=100, step=step, data=data
        )
        baseline_metrics.test_name = "execution_performance_baseline"
        self.results.baseline_metrics["execution_performance"] = baseline_metrics

        # Optimized
        optimized_metrics = await self.benchmark_function(
            self.optimized_executor.execute, iterations=100, step=step, data=data
        )
        optimized_metrics.test_name = "execution_performance_optimized"
        self.results.optimized_metrics["execution_performance"] = optimized_metrics

        # Calculate improvement
        improvement = (
            (baseline_metrics.mean_time - optimized_metrics.mean_time)
            / baseline_metrics.mean_time
            * 100
        )
        self.results.improvements["execution_performance"] = improvement

        print("Execution Performance:")
        print(f"  Baseline: {baseline_metrics.mean_time:.6f}s")
        print(f"  Optimized: {optimized_metrics.mean_time:.6f}s")
        print(f"  Improvement: {improvement:.1f}%")

    async def test_memory_efficiency(self) -> None:
        """Test memory usage efficiency."""
        step = self.create_test_step("memory_test")

        # Test with multiple executions to measure memory growth
        async def memory_test_baseline():
            for i in range(50):
                data = {"test": "memory", "iteration": i}
                await self.baseline_executor.execute(step, data)

        async def memory_test_optimized():
            for i in range(50):
                data = {"test": "memory", "iteration": i}
                await self.optimized_executor.execute(step, data)

        # Baseline
        baseline_metrics = await self.benchmark_function(
            memory_test_baseline, iterations=10, warmup=2
        )
        baseline_metrics.test_name = "memory_efficiency_baseline"
        self.results.baseline_metrics["memory_efficiency"] = baseline_metrics

        # Optimized
        optimized_metrics = await self.benchmark_function(
            memory_test_optimized, iterations=10, warmup=2
        )
        optimized_metrics.test_name = "memory_efficiency_optimized"
        self.results.optimized_metrics["memory_efficiency"] = optimized_metrics

        # Calculate improvement
        memory_improvement = (
            (baseline_metrics.memory_usage_mb - optimized_metrics.memory_usage_mb)
            / baseline_metrics.memory_usage_mb
            * 100
        )
        self.results.improvements["memory_efficiency"] = memory_improvement

        print("Memory Efficiency:")
        print(f"  Baseline: {baseline_metrics.memory_usage_mb:.2f} MB")
        print(f"  Optimized: {optimized_metrics.memory_usage_mb:.2f} MB")
        print(f"  Improvement: {memory_improvement:.1f}%")

    async def test_concurrent_performance(self) -> None:
        """Test concurrent execution performance."""
        step = self.create_test_step("concurrent_test")
        data = {"test": "concurrent"}

        async def concurrent_test_baseline():
            tasks = [self.baseline_executor.execute(step, data) for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        async def concurrent_test_optimized():
            tasks = [self.optimized_executor.execute(step, data) for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # Baseline
        baseline_metrics = await self.benchmark_function(
            concurrent_test_baseline, iterations=20, warmup=3
        )
        baseline_metrics.test_name = "concurrent_performance_baseline"
        self.results.baseline_metrics["concurrent_performance"] = baseline_metrics

        # Optimized
        optimized_metrics = await self.benchmark_function(
            concurrent_test_optimized, iterations=20, warmup=3
        )
        optimized_metrics.test_name = "concurrent_performance_optimized"
        self.results.optimized_metrics["concurrent_performance"] = optimized_metrics

        # Calculate improvement
        improvement = (
            (baseline_metrics.mean_time - optimized_metrics.mean_time)
            / baseline_metrics.mean_time
            * 100
        )
        self.results.improvements["concurrent_performance"] = improvement

        print("Concurrent Performance:")
        print(f"  Baseline: {baseline_metrics.mean_time:.6f}s")
        print(f"  Optimized: {optimized_metrics.mean_time:.6f}s")
        print(f"  Improvement: {improvement:.1f}%")

    async def test_cache_performance(self) -> None:
        """Test cache performance improvements."""
        step = self.create_test_step("cache_test")
        data = {"test": "cache", "value": 123}

        # Baseline cache test
        async def cache_test_baseline():
            # First execution (cache miss)
            await self.baseline_executor.execute(step, data)
            # Second execution (cache hit)
            await self.baseline_executor.execute(step, data)

        # Optimized cache test
        async def cache_test_optimized():
            # First execution (cache miss)
            await self.optimized_executor.execute(step, data)
            # Second execution (cache hit)
            await self.optimized_executor.execute(step, data)

        # Baseline
        baseline_metrics = await self.benchmark_function(
            cache_test_baseline, iterations=50, warmup=5
        )
        baseline_metrics.test_name = "cache_performance_baseline"
        self.results.baseline_metrics["cache_performance"] = baseline_metrics

        # Optimized
        optimized_metrics = await self.benchmark_function(
            cache_test_optimized, iterations=50, warmup=5
        )
        optimized_metrics.test_name = "cache_performance_optimized"
        self.results.optimized_metrics["cache_performance"] = optimized_metrics

        # Calculate improvement
        improvement = (
            (baseline_metrics.mean_time - optimized_metrics.mean_time)
            / baseline_metrics.mean_time
            * 100
        )
        self.results.improvements["cache_performance"] = improvement

        print("Cache Performance:")
        print(f"  Baseline: {baseline_metrics.mean_time:.6f}s")
        print(f"  Optimized: {optimized_metrics.mean_time:.6f}s")
        print(f"  Improvement: {improvement:.1f}%")

    async def test_context_handling_performance(self) -> None:
        """Test context handling performance."""
        step = self.create_test_step("context_test")
        data = {"test": "context"}

        # Create mock context
        mock_context = Mock()
        mock_context.model_dump.return_value = {"context": "test", "value": 456}

        # Baseline
        baseline_metrics = await self.benchmark_function(
            self.baseline_executor.execute,
            iterations=50,
            step=step,
            data=data,
            context=mock_context,
        )
        baseline_metrics.test_name = "context_handling_baseline"
        self.results.baseline_metrics["context_handling"] = baseline_metrics

        # Optimized
        optimized_metrics = await self.benchmark_function(
            self.optimized_executor.execute,
            iterations=50,
            step=step,
            data=data,
            context=mock_context,
        )
        optimized_metrics.test_name = "context_handling_optimized"
        self.results.optimized_metrics["context_handling"] = optimized_metrics

        # Calculate improvement
        improvement = (
            (baseline_metrics.mean_time - optimized_metrics.mean_time)
            / baseline_metrics.mean_time
            * 100
        )
        self.results.improvements["context_handling"] = improvement

        print("Context Handling Performance:")
        print(f"  Baseline: {baseline_metrics.mean_time:.6f}s")
        print(f"  Optimized: {optimized_metrics.mean_time:.6f}s")
        print(f"  Improvement: {improvement:.1f}%")

    def analyze_bottlenecks(self) -> None:
        """Analyze performance bottlenecks and identify opportunities."""
        bottlenecks = []
        recommendations = []

        # Check for performance regressions
        for test_name, improvement in self.results.improvements.items():
            if improvement < 0:
                bottlenecks.append(
                    f"{test_name}: Performance regression of {abs(improvement):.1f}%"
                )
                recommendations.append(
                    f"Investigate {test_name} optimization - may need parameter tuning"
                )

        # Check if improvements meet targets
        targets = {
            "execution_performance": 20.0,  # 20% improvement target
            "memory_efficiency": 30.0,  # 30% memory reduction target
            "concurrent_performance": 50.0,  # 50% concurrent improvement target
            "cache_performance": 25.0,  # 25% cache improvement target
            "context_handling": 40.0,  # 40% context handling improvement target
        }

        for test_name, target in targets.items():
            if test_name in self.results.improvements:
                actual = self.results.improvements[test_name]
                if actual < target:
                    bottlenecks.append(
                        f"{test_name}: Only {actual:.1f}% improvement vs {target:.1f}% target"
                    )
                    recommendations.append(
                        f"Tune {test_name} optimization parameters to reach {target:.1f}% target"
                    )

        # Check for high memory usage
        for test_name, metrics in self.results.optimized_metrics.items():
            if metrics.memory_usage_mb > 100:  # More than 100MB
                bottlenecks.append(
                    f"{test_name}: High memory usage ({metrics.memory_usage_mb:.1f} MB)"
                )
                recommendations.append(f"Optimize memory usage in {test_name}")

        # Check for high latency
        for test_name, metrics in self.results.optimized_metrics.items():
            if metrics.p99_time > 0.1:  # More than 100ms P99
                bottlenecks.append(f"{test_name}: High P99 latency ({metrics.p99_time:.3f}s)")
                recommendations.append(f"Optimize tail latency in {test_name}")

        self.results.bottlenecks = bottlenecks
        self.results.recommendations = recommendations

    def calculate_overall_score(self) -> None:
        """Calculate overall performance score."""
        # Weight different aspects of performance
        weights = {
            "execution_performance": 0.3,
            "memory_efficiency": 0.2,
            "concurrent_performance": 0.2,
            "cache_performance": 0.15,
            "context_handling": 0.15,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for test_name, weight in weights.items():
            if test_name in self.results.improvements:
                improvement = max(0, self.results.improvements[test_name])  # No negative scores
                weighted_score += improvement * weight
                total_weight += weight

        if total_weight > 0:
            self.results.overall_score = weighted_score / total_weight
        else:
            self.results.overall_score = 0.0

    async def run_validation(self) -> ValidationResults:
        """Run complete performance validation."""
        print("Starting Performance Validation...")
        print("=" * 50)

        # Run all performance tests
        await self.test_execution_performance()
        await self.test_memory_efficiency()
        await self.test_concurrent_performance()
        await self.test_cache_performance()
        await self.test_context_handling_performance()

        # Analyze results
        self.analyze_bottlenecks()
        self.calculate_overall_score()

        print("\n" + "=" * 50)
        print("Performance Validation Summary:")
        print(f"Overall Score: {self.results.overall_score:.1f}%")

        if self.results.bottlenecks:
            print("\nBottlenecks Identified:")
            for bottleneck in self.results.bottlenecks:
                print(f"  - {bottleneck}")

        if self.results.recommendations:
            print("\nRecommendations:")
            for recommendation in self.results.recommendations:
                print(f"  - {recommendation}")

        return self.results

    def save_results(self, filename: str = "performance_validation_results.json") -> None:
        """Save validation results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)
        print(f"\nResults saved to {filename}")


async def main():
    """Main validation function."""
    validator = PerformanceValidator()
    results = await validator.run_validation()
    validator.save_results()

    # Return exit code based on overall score
    if results.overall_score >= 20.0:  # At least 20% overall improvement
        print(f"\n✅ Performance validation PASSED (Score: {results.overall_score:.1f}%)")
        return 0
    else:
        print(f"\n❌ Performance validation FAILED (Score: {results.overall_score:.1f}%)")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
