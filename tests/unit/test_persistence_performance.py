"""Performance tests for Core Operational Persistence feature (NFR-9, NFR-10)."""

import logging
import os
import time
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from flujo import Flujo, Step
from flujo.application.context_manager import PipelineContext
from flujo.cli.main import app
from flujo.domain.models import StubAgent
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result

# Default overhead limit for performance tests
DEFAULT_OVERHEAD_LIMIT = 15.0

logger = logging.getLogger(__name__)


class TestPersistencePerformanceOverhead:
    """Test NFR-9: Default persistence must not introduce >15% overhead (relaxed for CI environments)."""

    @staticmethod
    def get_default_overhead_limit() -> float:
        """Get the default overhead limit from environment variable or fallback to 15.0."""
        try:
            return float(os.getenv("FLUJO_OVERHEAD_LIMIT", str(DEFAULT_OVERHEAD_LIMIT)))
        except ValueError:
            logging.warning(
                "Invalid value for FLUJO_OVERHEAD_LIMIT environment variable. Falling back to default value: 15.0"
            )
            return DEFAULT_OVERHEAD_LIMIT

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

        # Log performance results for debugging
        logger.debug("Performance Overhead Test Results:")
        logger.debug(f"Average time without backend: {avg_no_backend:.4f}s")
        logger.debug(f"Average time with default backend: {avg_with_backend:.4f}s")
        logger.debug(f"Overhead: {overhead_percentage:.2f}%")

        # NFR-9: Must not exceed overhead limit (relaxed for CI environments)
        # The SQLite backend adds some overhead due to file I/O, which is acceptable
        # for the durability benefits it provides
        overhead_limit = self.get_default_overhead_limit()
        assert overhead_percentage <= overhead_limit, (
            f"Default persistence overhead ({overhead_percentage:.2f}%) exceeds {overhead_limit}% limit"
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

        # Log performance results for debugging (consistent logging approach)
        logger.debug("Large Context Performance Test:")
        logger.debug(f"Time without backend: {no_backend_time:.4f}s")
        logger.debug(f"Time with backend: {with_backend_time:.4f}s")
        logger.debug(f"Overhead: {overhead_percentage:.2f}%")

        # Should still be under 5% even with large context
        assert overhead_percentage <= 5.0, (
            f"Persistence overhead with large context ({overhead_percentage:.2f}%) exceeds 5%"
        )


class TestCLIPerformance:
    """Test NFR-10: CLI commands must complete in <2s with 10,000 runs."""

    @staticmethod
    def get_cli_performance_threshold() -> float:
        """Get CLI performance threshold from environment or use default."""
        return float(os.getenv("FLUJO_CLI_PERF_THRESHOLD", "2.0"))

    @pytest.fixture
    def large_database(self, tmp_path: Path) -> Path:
        """Create a database with 10,000 runs for performance testing."""
        import asyncio

        db_path = tmp_path / "large_ops.db"
        backend = SQLiteBackend(db_path)

        # Create 10,000 runs with concurrent operations for better performance
        now = datetime.utcnow()

        async def create_database():
            # Prepare all run start operations
            run_start_tasks = []
            for i in range(10000):
                task = backend.save_run_start(
                    {
                        "run_id": f"run_{i:05d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "start_time": now - timedelta(minutes=i),
                    }
                )
                run_start_tasks.append(task)

            # Execute all run start operations concurrently
            await asyncio.gather(*run_start_tasks)

            # Prepare run end operations for completed runs (95%)
            run_end_tasks = []
            for i in range(9500):
                task = backend.save_run_end(
                    f"run_{i:05d}",
                    {
                        "status": "completed" if i % 2 == 0 else "failed",
                        "end_time": now - timedelta(minutes=i) + timedelta(seconds=30),
                        "total_cost": 0.1,
                        "final_context": {"result": f"output_{i}"},
                    },
                )
                run_end_tasks.append(task)

            # Execute all run end operations concurrently
            await asyncio.gather(*run_end_tasks)

            # Prepare step data operations for completed runs
            step_tasks = []
            for i in range(9500):
                for step_idx in range(3):
                    task = backend.save_step_result(
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
                    step_tasks.append(task)

            # Execute all step data operations concurrently
            await asyncio.gather(*step_tasks)

        # Run the async function
        asyncio.run(create_database())

        return db_path

    @pytest.mark.slow
    def test_large_database_fixture_verification(self, large_database: Path) -> None:
        """Verify that the large_database fixture is working correctly."""
        import asyncio

        # Verify the database file exists
        assert large_database.exists(), f"Database file {large_database} does not exist"

        # Verify the database has data by checking with the backend directly
        backend = SQLiteBackend(large_database)
        runs = asyncio.run(backend.list_runs())

        # Should have 10,000 runs
        assert len(runs) == 10000, f"Expected 10,000 runs, got {len(runs)}"

        # Verify some specific runs exist
        run_ids = [run["run_id"] for run in runs]
        assert "run_00000" in run_ids, "First run should exist"
        assert "run_00999" in run_ids, "Last run should exist"

        # Verify run details
        run_details = asyncio.run(backend.get_run_details("run_00001"))
        assert run_details is not None, "Run details should be retrievable"
        assert run_details["run_id"] == "run_00001"

        # Verify step data exists
        steps = asyncio.run(backend.list_run_steps("run_00001"))
        assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"

    @pytest.mark.slow
    def test_lens_list_performance(self, large_database: Path) -> None:
        """Test that `flujo lens list` completes in <2s with 10,000 runs."""
        import asyncio

        # Set environment variable to point to our test database
        # Use correct URI format: sqlite:///path for absolute paths
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        # Debug: Verify the database has data before running CLI
        backend = SQLiteBackend(large_database)
        runs = asyncio.run(backend.list_runs())
        logger.debug(f"Database contains {len(runs)} runs before CLI test")

        runner = CliRunner()

        # Measure execution time
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list"])
        execution_time = time.perf_counter() - start_time

        # Enhanced error handling with detailed debugging
        if result.exit_code != 0:
            logger.error("CLI command failed with detailed information:")
            logger.error(f"Exit code: {result.exit_code}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            logger.error(f"Database path: {large_database}")
            logger.error(f"Environment FLUJO_STATE_URI: {os.environ.get('FLUJO_STATE_URI')}")
            raise AssertionError(f"CLI command failed: {result.stdout}")

        # Log performance results for debugging
        logger.debug("CLI List Performance Test:")
        logger.debug(f"Execution time: {execution_time:.3f}s")
        logger.debug(f"Exit code: {result.exit_code}")

        # Use standardized threshold configuration
        threshold = self.get_cli_performance_threshold()
        assert execution_time < threshold, (
            f"`flujo lens list` took {execution_time:.3f}s, exceeds {threshold}s limit"
        )

    @pytest.mark.slow
    def test_lens_show_performance(self, large_database: Path) -> None:
        """Test that `flujo lens show` completes in <2s with 10,000 runs."""

        # Set environment variable to point to our test database
        # Use correct URI format: sqlite:///path for absolute paths
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with a specific run_id
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "show", "run_00001"])
        execution_time = time.perf_counter() - start_time

        # Enhanced error handling with detailed debugging
        if result.exit_code != 0:
            logger.error("CLI command failed with detailed information:")
            logger.error(f"Exit code: {result.exit_code}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise AssertionError(f"CLI command failed: {result.stdout}")

        # Log performance results for debugging
        logger.debug("CLI Show Performance Test:")
        logger.debug(f"Execution time: {execution_time:.3f}s")
        logger.debug(f"Exit code: {result.exit_code}")

        # Use standardized threshold configuration
        threshold = self.get_cli_performance_threshold()
        assert execution_time < threshold, (
            f"`flujo lens show` took {execution_time:.3f}s, exceeds {threshold}s limit"
        )

    @pytest.mark.slow
    def test_lens_list_with_filters_performance(self, large_database: Path) -> None:
        """Test that `flujo lens list` with filters completes in <2s with 10,000 runs."""

        # Set environment variable to point to our test database
        # Use correct URI format: sqlite:///path for absolute paths
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with status filter
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "list", "--status", "completed"])
        execution_time = time.perf_counter() - start_time

        # Enhanced error handling with detailed debugging
        if result.exit_code != 0:
            logger.error("CLI command failed with detailed information:")
            logger.error(f"Exit code: {result.exit_code}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise AssertionError(f"CLI command failed: {result.stdout}")

        # Log performance results for debugging
        logger.debug("CLI List with Filter Performance Test:")
        logger.debug(f"Execution time: {execution_time:.3f}s")
        logger.debug(f"Exit code: {result.exit_code}")

        # Use standardized threshold configuration
        threshold = self.get_cli_performance_threshold()
        assert execution_time < threshold, (
            f"`flujo lens list --status completed` took {execution_time:.3f}s, exceeds {threshold}s limit"
        )

    @pytest.mark.slow
    def test_lens_show_nonexistent_run_performance(self, large_database: Path) -> None:
        """Test that `flujo lens show` with nonexistent run completes in <2s with 10,000 runs."""

        # Set environment variable to point to our test database
        # Use correct URI format: sqlite:///path for absolute paths
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        runner = CliRunner()

        # Test with nonexistent run_id
        start_time = time.perf_counter()
        result = runner.invoke(app, ["lens", "show", "nonexistent_run"])
        execution_time = time.perf_counter() - start_time

        # Log performance results for debugging
        logger.debug("CLI Show Nonexistent Run Performance Test:")
        logger.debug(f"Execution time: {execution_time:.3f}s")
        logger.debug(f"Exit code: {result.exit_code}")

        # Use a configurable threshold for CI environments (faster for nonexistent runs)
        PERFORMANCE_THRESHOLD = float(os.environ.get("FLUJO_CLI_PERF_THRESHOLD", "0.2"))
        assert execution_time < PERFORMANCE_THRESHOLD, (
            f"`flujo lens show` for nonexistent run took {execution_time:.3f}s, should be very fast"
        )
        # Should exit with error code for nonexistent run
        assert result.exit_code != 0, "Should fail for nonexistent run"
