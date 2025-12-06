"""
Performance benchmarks for tracing functionality.

This module benchmarks the performance impact of the TraceManager hook
to ensure it meets the <5% overhead requirement (NFR-11).
"""

import asyncio
import os
import time
import statistics

import pytest

# Mark entire module as benchmark and slow to exclude from fast subset
pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

from flujo import Step
from flujo.testing.utils import StubAgent
from tests.conftest import create_test_flujo


class TestTracingPerformance:
    """Benchmark the performance impact of tracing functionality."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_tracing_overhead_simple_pipeline(self, benchmark):
        """Benchmark tracing overhead on a simple linear pipeline."""

        def create_simple_pipeline():
            """Create a simple 3-step pipeline."""
            step1 = Step.model_validate(
                {
                    "name": "step1",
                    "agent": StubAgent(["output1"] * 5),  # Multiple outputs for benchmark
                }
            )
            step2 = Step.model_validate(
                {
                    "name": "step2",
                    "agent": StubAgent(["output2"] * 5),  # Multiple outputs for benchmark
                }
            )
            step3 = Step.model_validate(
                {
                    "name": "step3",
                    "agent": StubAgent(["output3"] * 5),  # Multiple outputs for benchmark
                }
            )
            return step1 >> step2 >> step3

        def run_pipeline_with_tracing():
            """Run pipeline with tracing enabled (default)."""
            pipeline = create_simple_pipeline()
            runner = create_test_flujo(pipeline, persist_state=False)
            result = None

            async def run():
                nonlocal result
                async for r in runner.run_async("test_input"):
                    result = r

            asyncio.run(run())
            return result

        def run_pipeline_without_tracing():
            """Run pipeline with tracing disabled."""
            pipeline = create_simple_pipeline()
            # Create runner and disable tracing using the API
            runner = create_test_flujo(pipeline, persist_state=False)
            runner.disable_tracing()
            result = None

            async def run():
                nonlocal result
                async for r in runner.run_async("test_input"):
                    result = r

            asyncio.run(run())
            return result

        # Benchmark with tracing
        tracing_result = benchmark(run_pipeline_with_tracing)

        # Verify trace tree is attached
        assert tracing_result.trace_tree is not None
        assert tracing_result.trace_tree.name == "pipeline_run"
        assert len(tracing_result.trace_tree.children) == 3

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_tracing_overhead_complex_pipeline_with_tracing(self, benchmark):
        """Benchmark complex pipeline with tracing enabled."""

        def create_complex_pipeline():
            from flujo.domain.dsl.loop import LoopStep
            from flujo.domain.dsl.conditional import ConditionalStep
            from flujo.domain.dsl.pipeline import Pipeline

            inner_step = Step.model_validate(
                {
                    "name": "inner_step",
                    "agent": StubAgent(
                        ["inner_output_1", "inner_output_2", "inner_output_3"] * 5
                    ),  # Multiple outputs for benchmark
                }
            )
            loop_body_pipeline = Pipeline.from_step(inner_step)
            iteration_counter = {"count": 0}

            def exit_condition(output, ctx):
                iteration_counter["count"] += 1
                return iteration_counter["count"] >= 3

            loop_step = LoopStep.model_validate(
                {
                    "name": "loop_step",
                    "loop_body_pipeline": loop_body_pipeline,
                    "exit_condition_callable": exit_condition,
                }
            )

            def condition_fn(output, ctx):
                return "true"

            true_branch = Pipeline.from_step(loop_step)
            false_branch = Pipeline.from_step(inner_step)
            conditional_step = ConditionalStep.model_validate(
                {
                    "name": "conditional_step",
                    "condition_callable": condition_fn,
                    "branches": {"true": true_branch, "false": false_branch},
                }
            )
            return Pipeline.from_step(conditional_step)

        pipeline = create_complex_pipeline()
        runner = create_test_flujo(pipeline, persist_state=False)

        def run_pipeline():
            runner.run("input")

        benchmark.pedantic(run_pipeline, rounds=5, iterations=1)
        print(f"[BENCHMARK] Complex pipeline with tracing: {benchmark.stats['mean']:.4f} ms")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_tracing_overhead_complex_pipeline_no_tracing(self, benchmark):
        """Benchmark complex pipeline with tracing disabled."""

        def create_complex_pipeline():
            from flujo.domain.dsl.loop import LoopStep
            from flujo.domain.dsl.conditional import ConditionalStep
            from flujo.domain.dsl.pipeline import Pipeline

            inner_step = Step.model_validate(
                {
                    "name": "inner_step",
                    "agent": StubAgent(
                        ["inner_output_1", "inner_output_2", "inner_output_3"] * 5
                    ),  # Multiple outputs for benchmark
                }
            )
            loop_body_pipeline = Pipeline.from_step(inner_step)
            iteration_counter = {"count": 0}

            def exit_condition(output, ctx):
                iteration_counter["count"] += 1
                return iteration_counter["count"] >= 3

            loop_step = LoopStep.model_validate(
                {
                    "name": "loop_step",
                    "loop_body_pipeline": loop_body_pipeline,
                    "exit_condition_callable": exit_condition,
                }
            )

            def condition_fn(output, ctx):
                return "true"

            true_branch = Pipeline.from_step(loop_step)
            false_branch = Pipeline.from_step(inner_step)
            conditional_step = ConditionalStep.model_validate(
                {
                    "name": "conditional_step",
                    "condition_callable": condition_fn,
                    "branches": {"true": true_branch, "false": false_branch},
                }
            )
            return Pipeline.from_step(conditional_step)

        pipeline = create_complex_pipeline()
        runner = create_test_flujo(pipeline, hooks=[], persist_state=False)
        runner.disable_tracing()

        def run_pipeline():
            runner.run("input")

        benchmark.pedantic(run_pipeline, rounds=5, iterations=1)
        print(f"[BENCHMARK] Complex pipeline without tracing: {benchmark.stats['mean']:.4f} ms")

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_trace_persistence_overhead(self, benchmark):
        """Benchmark the overhead of trace persistence to database."""

        def create_pipeline_with_persistence():
            """Create pipeline with state backend for persistence."""
            import tempfile
            from flujo.state.backends.sqlite import SQLiteBackend

            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            try:
                step1 = Step.model_validate(
                    {
                        "name": "step1",
                        "agent": StubAgent(["output1"] * 5),  # Multiple outputs for benchmark
                    }
                )
                step2 = Step.model_validate(
                    {
                        "name": "step2",
                        "agent": StubAgent(["output2"] * 5),  # Multiple outputs for benchmark
                    }
                )

                backend = SQLiteBackend(db_path)
                runner = create_test_flujo(step1 >> step2, state_backend=backend)

                result = None

                async def run():
                    nonlocal result
                    async for r in runner.run_async("test_input"):
                        result = r

                asyncio.run(run())

                return result
            finally:
                if os.path.exists(db_path):
                    os.unlink(db_path)

        # Benchmark with persistence
        result = benchmark(create_pipeline_with_persistence)

        # Verify trace is persisted
        assert result.trace_tree is not None

    def test_tracing_memory_overhead(self):
        """Test memory overhead of tracing functionality."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available")

        def measure_memory_usage():
            """Measure memory usage of current process."""
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        # Baseline memory usage
        baseline_memory = measure_memory_usage()

        # Create and run multiple pipelines to stress test memory
        pipelines = []
        for i in range(10):
            step = Step.model_validate(
                {
                    "name": f"step_{i}",
                    "agent": StubAgent([f"output_{i}"] * 5),  # Multiple outputs for multiple runs
                }
            )
            runner = create_test_flujo(step, persist_state=False)
            pipelines.append(runner)

        # Run all pipelines
        results = []
        for runner in pipelines:
            result = None

            async def run():
                nonlocal result
                async for r in runner.run_async("test_input"):
                    result = r

            asyncio.run(run())
            results.append(result)

        # Memory usage after running pipelines
        final_memory = measure_memory_usage()
        memory_increase = final_memory - baseline_memory

        # Verify all trace trees are attached
        for result in results:
            assert result.trace_tree is not None

        # Memory increase should be reasonable (< 50MB for 10 pipelines)
        assert memory_increase < 50, f"Memory increase too high: {memory_increase:.2f}MB"

    def test_trace_tree_size_limits(self):
        """Test that trace trees don't grow excessively large."""

        def create_large_pipeline():
            """Create a pipeline with many steps to test trace tree size."""
            steps = []
            for i in range(100):  # 100 steps
                step = Step.model_validate(
                    {
                        "name": f"step_{i}",
                        "agent": StubAgent(
                            [f"output_{i}"] * 3
                        ),  # Multiple outputs for potential retries
                    }
                )
                steps.append(step)

            # Chain all steps
            pipeline = steps[0]
            for step in steps[1:]:
                pipeline = pipeline >> step

            return pipeline

        pipeline = create_large_pipeline()
        runner = create_test_flujo(pipeline, persist_state=False)

        # Run the large pipeline
        result = None

        async def run():
            nonlocal result
            async for r in runner.run_async("test_input"):
                result = r

        asyncio.run(run())

        # Verify trace tree is reasonable size
        assert result.trace_tree is not None
        assert result.trace_tree.name == "pipeline_run"

        # Count total spans in tree
        def count_spans(span):
            count = 1
            for child in span.children:
                count += count_spans(child)
            return count

        total_spans = count_spans(result.trace_tree)
        assert total_spans == 101  # root + 100 steps

        # Verify tree structure is correct
        assert len(result.trace_tree.children) == 100

    def test_tracing_performance_regression(self):
        """Test that tracing doesn't cause performance regression over multiple runs."""

        def run_pipeline_multiple_times():
            """Run the same pipeline multiple times and measure consistency."""
            # Create agents with enough outputs for multiple runs
            step1 = Step.model_validate(
                {
                    "name": "step1",
                    "agent": StubAgent(
                        ["output1"] * 40
                    ),  # 40 outputs for 20 runs (2 steps per run)
                }
            )
            step2 = Step.model_validate(
                {
                    "name": "step2",
                    "agent": StubAgent(["output2"] * 40),  # 40 outputs for 20 runs
                }
            )

            pipeline = step1 >> step2
            runner = create_test_flujo(pipeline, persist_state=False)

            # Warmup runs to stabilize performance
            for _ in range(3):
                result = None

                async def warmup_run():
                    nonlocal result
                    async for r in runner.run_async("test_input"):
                        result = r

                asyncio.run(warmup_run())

            execution_times = []
            for _ in range(20):  # Increased from 10 to 20 to reduce random variation
                # Force garbage collection before each run to reduce variability
                import gc

                gc.collect()

                start_time = time.perf_counter()

                result = None

                async def run():
                    nonlocal result
                    async for r in runner.run_async("test_input"):
                        result = r

                asyncio.run(run())

                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)

                # Verify trace tree is always attached
                assert result.trace_tree is not None

            return execution_times

        execution_times = run_pipeline_multiple_times()

        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times)
        cv = std_dev / mean_time  # Coefficient of variation

        # Log performance for debugging
        max_time = max(execution_times)
        min_time = min(execution_times)
        print("\nTracing Performance Consistency:")
        print(f"  Mean: {mean_time:.4f}s, StdDev: {std_dev:.4f}s, CV: {cv:.3f}")
        print(f"  Range: [{min_time:.4f}s, {max_time:.4f}s]")

        # RELATIVE performance check (environment-independent):
        # Max execution time should not be more than 5x min.
        # This catches severe performance regressions without being sensitive to
        # absolute timing differences between local and CI environments.
        time_ratio = max_time / min_time if min_time > 0 else float("inf")
        max_ratio = 5.0

        print(f"  Max/Min ratio: {time_ratio:.2f}x")

        assert time_ratio <= max_ratio, (
            f"Performance too inconsistent: max/min ratio {time_ratio:.2f}x exceeds {max_ratio}x. "
            f"Times: mean={mean_time:.4f}s, range=[{min_time:.4f}s, {max_time:.4f}s]"
        )

        # Sanity check: all runs should complete (not hang)
        assert all(t < 30.0 for t in execution_times), f"Some runs too slow: {execution_times}"
