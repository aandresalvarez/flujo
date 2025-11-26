#!/usr/bin/env python3
"""Baseline measurement script for Flujo performance tests.

This script runs performance measurements to establish or update baselines
for the robustness test suite.
"""
# ruff: noqa

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.robustness.baseline_manager import get_baseline_manager, measure_and_check_regression
from tests.test_types.mocks import create_mock_executor_core
from flujo.domain.dsl.step import Step
from unittest.mock import AsyncMock


async def measure_step_execution():
    """Measure step execution performance."""
    print("📊 Measuring step execution performance...")

    manager = get_baseline_manager()

    async def single_measurement():
        executor = create_mock_executor_core()
        step = Step(name="benchmark_step", agent=AsyncMock())
        result = await executor.execute(step, {"input": "benchmark"})
        return result

    # Measure multiple iterations
    times = []
    for _ in range(20):
        start_time = time.perf_counter()
        await single_measurement()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    manager.add_measurement("step_execution", avg_time)
    is_regression, message = manager.check_regression("step_execution", avg_time)

    print(f"   Result: {avg_time:.2f}ms ± {std_dev:.2f}ms")
    print(f"   Status: {'⚠️  REGRESSION' if is_regression else '✅ OK'}")
    print(f"   Details: {message}")
    print()


async def measure_cache_performance():
    """Measure cache performance improvement."""
    print("📊 Measuring cache performance improvement...")

    manager = get_baseline_manager()

    async def measure_single_cache_ratio():
        executor = create_mock_executor_core(cache_hit=True)
        step = Step(name="cached_benchmark_step", agent=AsyncMock())

        # First execution (cache miss)
        start1 = time.perf_counter()
        await executor.execute(step, {"input": "cache_test"})
        time1 = (time.perf_counter() - start1) * 1000

        # Second execution (cache hit)
        start2 = time.perf_counter()
        await executor.execute(step, {"input": "cache_test"})
        time2 = (time.perf_counter() - start2) * 1000

        ratio = time1 / time2 if time2 > 0 else float("inf")
        return ratio

    # Measure multiple cache ratios
    ratios = []
    for _ in range(10):
        ratio = await measure_single_cache_ratio()
        ratios.append(ratio)

    avg_ratio = sum(ratios) / len(ratios)
    std_dev = (sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)) ** 0.5

    manager.add_measurement("cache_improvement_ratio", avg_ratio)
    is_regression, message = manager.check_regression("cache_improvement_ratio", avg_ratio)

    print(f"   Result: {avg_ratio:.2f}x improvement ± {std_dev:.2f}")
    print(f"   Status: {'⚠️  REGRESSION' if is_regression else '✅ OK'}")
    print(f"   Details: {message}")
    print()


async def measure_concurrent_execution():
    """Measure concurrent execution performance."""
    print("📊 Measuring concurrent execution performance...")

    manager = get_baseline_manager()

    async def execute_concurrent_operations():
        executor = create_mock_executor_core()

        async def single_operation(idx):
            step = Step(name=f"concurrent_step_{idx}", agent=AsyncMock())
            return await executor.execute(step, {"input": f"test_{idx}"})

        # Execute 10 concurrent operations
        tasks = [single_operation(i) for i in range(10)]
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000  # ms
        return total_time

    # Measure multiple concurrent executions
    times = []
    for _ in range(15):
        total_time = await execute_concurrent_operations()
        times.append(total_time)

    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    manager.add_measurement("concurrent_execution", avg_time)
    is_regression, message = manager.check_regression("concurrent_execution", avg_time)

    print(f"   Result: {avg_time:.2f}ms ± {std_dev:.2f}ms")
    print(f"   Status: {'⚠️  REGRESSION' if is_regression else '✅ OK'}")
    print(f"   Details: {message}")
    print()


def measure_pipeline_creation():
    """Measure pipeline creation performance."""
    print("📊 Measuring pipeline creation performance...")

    def create_pipeline():
        from flujo.domain.dsl.pipeline import Pipeline

        steps = [Step(name=f"step_{i}", agent=AsyncMock()) for i in range(10)]
        pipeline = Pipeline(name="benchmark_pipeline", steps=steps)
        return pipeline

    manager = get_baseline_manager()
    avg_time, std_dev = manager.measure_performance(
        create_pipeline, "pipeline_creation", iterations=25
    )

    is_regression, message = manager.check_regression("pipeline_creation", avg_time)

    print(f"   Result: {avg_time:.2f}ms ± {std_dev:.2f}ms")
    print(f"   Status: {'⚠️  REGRESSION' if is_regression else '✅ OK'}")
    print(f"   Details: {message}")
    print()


async def run_baseline_measurements():
    """Run all baseline measurements."""
    print("🚀 Starting Flujo Performance Baseline Measurements")
    print("=" * 60)
    print()

    try:
        # Measure each performance aspect
        await measure_step_execution()
        await measure_cache_performance()
        await measure_concurrent_execution()
        measure_pipeline_creation()

        # Save updated baselines
        manager = get_baseline_manager()
        manager.save_baselines()

        print("✅ Baseline measurements completed and saved!")
        print()
        print("📋 Summary:")
        print(manager.get_baseline_summary())

    except Exception as e:
        print(f"❌ Error during baseline measurement: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def main():
    """Main entry point."""
    return asyncio.run(run_baseline_measurements())


if __name__ == "__main__":
    sys.exit(main())
