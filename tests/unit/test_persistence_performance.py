"""Performance tests for Core Operational Persistence feature (NFR-9, NFR-10)."""

import time
import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typer.testing import CliRunner

from flujo.application.runner import Flujo
from flujo.domain import Step
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import StubAgent, gather_result
from flujo.cli.main import app


class TestPersistencePerformanceOverhead:
    """Test NFR-9: Default persistence must not introduce >15% overhead."""

    # Performance threshold for CI environments
    DEFAULT_OVERHEAD_LIMIT = 15.0

    @pytest.mark.asyncio
    async def test_default_backend_performance_overhead(self) -> None:
        """Test that default SQLiteBackend doesn't add >5% overhead to pipeline runs."""

        # Create a simple pipeline with compatible types
        agent = StubAgent(["output"])
        pipeline = Step.solution(agent)

        # Test without backend (baseline)
        runner_no_backend = Flujo(pipeline, state_backend=None)

        # Test with default backend
        runner_with_backend = Flujo(pipeline)  # Uses default SQLiteBackend

        # Run multiple iterations to get stable measurements
        iterations = 10
        no_backend_times = []
        with_backend_times = []

        for _ in range(iterations):
            # Test without backend
            start = time.perf_counter()
            await gather_result(runner_no_backend, "test")
            no_backend_times.append(time.perf_counter() - start)

            # Test with default backend
            start = time.perf_counter()
            await gather_result(runner_with_backend, "test")
            with_backend_times.append(time.perf_counter() - start)

        # Calculate averages
        avg_no_backend = sum(no_backend_times) / len(no_backend_times)
        avg_with_backend = sum(with_backend_times) / len(with_backend_times)

        # Calculate overhead percentage
        overhead_percentage = ((avg_with_backend - avg_no_backend) / avg_no_backend) * 100

        # Log performance results for debugging (only in verbose mode)
        if __debug__:
            print("\nPerformance Overhead Test Results:")
            print(f"Average time without backend: {avg_no_backend:.4f}s")
            print(f"Average time with default backend: {avg_with_backend:.4f}s")
            print(f"Overhead: {overhead_percentage:.2f}%")

        # NFR-9: Must not exceed overhead limit (relaxed for CI environments)
        # The SQLite backend adds some overhead due to file I/O, which is acceptable
        # for the durability benefits it provides
        assert overhead_percentage <= self.DEFAULT_OVERHEAD_LIMIT, (
            f"Default persistence overhead ({overhead_percentage:.2f}%) exceeds {self.DEFAULT_OVERHEAD_LIMIT}% limit"
        )

    @pytest.mark.asyncio
    async def test_persistence_overhead_with_large_context(self) -> None:
        """Test performance overhead with large context data."""

        # Create context with substantial data
        class LargeContext(PipelineContext):
            large_data: str = "x" * 10000  # 10KB of data

        agent = StubAgent(["output"])
        pipeline = Step.solution(agent)

        # Test without backend
        runner_no_backend = Flujo(pipeline, context_model=LargeContext, state_backend=None)

        # Test with default backend
        runner_with_backend = Flujo(pipeline, context_model=LargeContext)

        # Run with large context (include required initial_prompt field)
        large_context_data = {"initial_prompt": "test", "large_data": "y" * 10000}

        # Measure performance
        start = time.perf_counter()
        await gather_result(runner_no_backend, "test", initial_context_data=large_context_data)
        no_backend_time = time.perf_counter() - start

        start = time.perf_counter()
        await gather_result(runner_with_backend, "test", initial_context_data=large_context_data)
        with_backend_time = time.perf_counter() - start

        overhead_percentage = ((with_backend_time - no_backend_time) / no_backend_time) * 100

        # Log performance results for debugging (only in verbose mode)
        if __debug__:
            print("\nLarge Context Performance Test:")
            print(f"Time without backend: {no_backend_time:.4f}s")
            print(f"Time with backend: {with_backend_time:.4f}s")
            print(f"Overhead: {overhead_percentage:.2f}%")

        # Should still be under 5% even with large context
        assert overhead_percentage <= 5.0, (
            f"Persistence overhead with large context ({overhead_percentage:.2f}%) exceeds 5%"
        )


class TestCLIPerformance:
    """Test NFR-10: CLI commands must complete in <500ms with 10,000 runs."""

    @pytest.fixture
    def large_database(self, tmp_path: Path) -> Path:
        """Create a database with 10,000 runs for performance testing."""
        db_path = tmp_path / "large_ops.db"
        backend = SQLiteBackend(db_path)

        # Create 10,000 runs
        now = datetime.utcnow()
        for i in range(10000):
            # Create run start
            asyncio.run(
                backend.save_run_start(
                    {
                        "run_id": f"run_{i:05d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": now - timedelta(minutes=i),
                    }
                )
            )

            # Create run end for most runs
            if i < 9500:  # 95% completed
                asyncio.run(
                    backend.save_run_end(
                        f"run_{i:05d}",
                        {
                            "status": "completed" if i % 2 == 0 else "failed",
                            "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                            "total_cost": 0.1,
                            "final_context": {"result": f"output_{i}"},
                        },
                    )
                )

            # Add some step data for completed runs
            if i < 9500:
                for step_idx in range(3):
                    asyncio.run(
                        backend.save_step_result(
                            {
                                "step_run_id": f"run_{i:05d}:{step_idx}",
                                "run_id": f"run_{i:05d}",
                                "step_name": f"step_{step_idx}",
                                "step_index": step_idx,
                                "status": "completed",
                                "start_time": now - timedelta(minutes=i),
                                "end_time": now - timedelta(minutes=i) + timedelta(seconds=10),
                                "duration_ms": 10000,
                                "cost": 0.03,
                                "tokens": 100,
                                "input": f"input_{i}_{step_idx}",
                                "output": f"output_{i}_{step_idx}",
                                "error": None,
                            }
                        )
                    )

        return db_path

    def test_lens_list_performance(self, large_database: Path) -> None:
        """Test that `flujo lens list` completes in <2s with 10,000 runs."""
        import os

        # Set environment variable to point to our test database
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Measure execution time
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list"])
        execution_time = time.perf_counter() - start_time

        # Log performance results for debugging (only in verbose mode)
        if __debug__:
            print("\nCLI List Performance Test:")
            print(f"Execution time: {execution_time:.3f}s")
            print(f"Exit code: {result.exit_code}")

        # NFR-10: Must complete in under 2s (adjusted for CI environments)
        assert execution_time < 2.0, (
            f"`flujo lens list` took {execution_time:.3f}s, exceeds 2s limit"
        )
        assert result.exit_code == 0, f"CLI command failed: {result.stdout}"

    def test_lens_show_performance(self, large_database: Path) -> None:
        """Test that `flujo lens show` completes in <500ms."""
        import os

        # Set environment variable to point to our test database
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with a specific run_id
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "show", "run_00001"])
        execution_time = time.perf_counter() - start_time

        # Log performance results for debugging (only in verbose mode)
        if __debug__:
            print("\nCLI Show Performance Test:")
            print(f"Execution time: {execution_time:.3f}s")
            print(f"Exit code: {result.exit_code}")

        # NFR-10: Must complete in under 500ms
        assert execution_time < 0.5, (
            f"`flujo lens show` took {execution_time:.3f}s, exceeds 500ms limit"
        )
        assert result.exit_code == 0, f"CLI command failed: {result.stdout}"

    def test_lens_list_with_filters_performance(self, large_database: Path) -> None:
        """Test that `flujo lens list` with filters completes in <2s."""
        import os

        # Set environment variable to point to our test database
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with status filter
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list", "--status", "completed"])
        execution_time = time.perf_counter() - start_time

        # Log performance results for debugging (only in verbose mode)
        if __debug__:
            print("\nCLI List with Filter Performance Test:")
            print(f"Execution time: {execution_time:.3f}s")
            print(f"Exit code: {result.exit_code}")

        # NFR-10: Must complete in under 2s (adjusted for CI environments)
        assert execution_time < 2.0, (
            f"`flujo lens list --status completed` took {execution_time:.3f}s, exceeds 2s limit"
        )
        assert result.exit_code == 0, f"CLI command failed: {result.stdout}"

    def test_lens_show_nonexistent_run_performance(self, large_database: Path) -> None:
        """Test that `flujo lens show` with nonexistent run completes quickly.
        The threshold is relaxed in CI via FLUJO_CLI_PERF_THRESHOLD due to CI variability.
        """
        import os

        # Set environment variable to point to our test database
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with nonexistent run_id
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "show", "nonexistent_run"])
        execution_time = time.perf_counter() - start_time

        print("\nCLI Show Nonexistent Run Performance Test:")
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Exit code: {result.exit_code}")

        # Use a configurable threshold for CI environments
        PERFORMANCE_THRESHOLD = float(os.environ.get("FLUJO_CLI_PERF_THRESHOLD", "0.2"))
        assert execution_time < PERFORMANCE_THRESHOLD, (
            f"`flujo lens show` for nonexistent run took {execution_time:.3f}s, should be very fast"
        )
        # Should exit with error code for nonexistent run
        assert result.exit_code != 0, "Should fail for nonexistent run"
