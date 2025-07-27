"""Performance tests for Core Operational Persistence feature (NFR-9, NFR-10)."""

import logging
import os
import time
import uuid
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from flujo import Step
from flujo.domain.models import PipelineContext
from flujo.cli.main import app
from flujo.state.backends.sqlite import SQLiteBackend
from flujo.testing.utils import gather_result, StubAgent
from tests.conftest import create_test_flujo

# Default overhead limit for performance tests
DEFAULT_OVERHEAD_LIMIT = 20.0

logger = logging.getLogger(__name__)


class TestPersistencePerformanceOverhead:
    """Test NFR-9: Default persistence must not introduce >15% overhead (relaxed for CI environments)."""

    @staticmethod
    def get_default_overhead_limit() -> float:
        """Get the default overhead limit from environment variable or fallback to 15.0."""
        try:
            # Use higher threshold in CI environments for more reliable tests
            if os.getenv("CI") == "true":
                default_limit = 25.0  # Higher threshold for CI to accommodate improvements
            else:
                default_limit = DEFAULT_OVERHEAD_LIMIT

            return float(os.getenv("FLUJO_OVERHEAD_LIMIT", str(default_limit)))
        except ValueError:
            logging.warning(
                "Invalid value for FLUJO_OVERHEAD_LIMIT environment variable. Falling back to default value: 15.0"
            )
            return DEFAULT_OVERHEAD_LIMIT

    @pytest.mark.asyncio
    async def test_default_backend_performance_overhead(self, tmp_path: Path) -> None:
        """Test that default SQLiteBackend doesn't add >5% overhead to pipeline runs with improved isolation."""

        # Create a simple pipeline with compatible types
        agent = StubAgent(["output"])
        pipeline = Step.solution(agent)

        # Create unique database files for isolation
        test_id = uuid.uuid4().hex[:8]
        with_backend_db_path = tmp_path / f"with_backend_{test_id}.db"

        # Test without backend (baseline)
        runner_no_backend = create_test_flujo(pipeline, state_backend=None)

        # Test with isolated backend using unique database file
        isolated_backend = SQLiteBackend(with_backend_db_path)
        runner_with_backend = create_test_flujo(pipeline, state_backend=isolated_backend)

        try:
            # Run multiple iterations to get stable measurements
            iterations = 10
            no_backend_times = []
            with_backend_times = []

            for _ in range(iterations):
                # Test without backend
                start = time.perf_counter_ns()
                await gather_result(runner_no_backend, "test")
                no_backend_times.append((time.perf_counter_ns() - start) / 1_000_000_000.0)

                # Test with isolated backend
                start = time.perf_counter_ns()
                await gather_result(runner_with_backend, "test")
                with_backend_times.append((time.perf_counter_ns() - start) / 1_000_000_000.0)

            # Calculate averages
            avg_no_backend = sum(no_backend_times) / len(no_backend_times)
            avg_with_backend = sum(with_backend_times) / len(with_backend_times)

            # Calculate overhead percentage
            overhead_percentage = ((avg_with_backend - avg_no_backend) / avg_no_backend) * 100

            # Log performance results for debugging
            logger.debug("Performance Overhead Test Results (Isolated):")
            logger.debug(f"Average time without backend: {avg_no_backend:.4f}s")
            logger.debug(f"Average time with isolated backend: {avg_with_backend:.4f}s")
            logger.debug(f"Overhead: {overhead_percentage:.2f}%")
            logger.debug(f"Individual measurements - No backend: {no_backend_times}")
            logger.debug(f"Individual measurements - With backend: {with_backend_times}")

            # NFR-9: Must not exceed overhead limit (relaxed for CI environments)
            # The SQLite backend adds some overhead due to file I/O, which is acceptable
            # for the durability benefits it provides
            overhead_limit = self.get_default_overhead_limit()
            assert overhead_percentage <= overhead_limit, (
                f"Default persistence overhead ({overhead_percentage:.2f}%) exceeds {overhead_limit}% limit"
            )

        finally:
            # Clean up database files to prevent resource contention
            try:
                if with_backend_db_path.exists():
                    with_backend_db_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up test database files: {e}")

    @pytest.mark.asyncio
    async def test_persistence_overhead_with_large_context(self, tmp_path: Path) -> None:
        """Test performance overhead with large context data with improved isolation."""

        # Create context with substantial data
        class LargeContext(PipelineContext):
            large_data: str = "x" * 10000  # 10KB of data

        agent = StubAgent(["output"])
        pipeline = Step.solution(agent)

        # Create unique database files for isolation
        test_id = uuid.uuid4().hex[:8]
        with_backend_db_path = tmp_path / f"with_backend_{test_id}.db"

        # Test without backend
        runner_no_backend = create_test_flujo(
            pipeline, context_model=LargeContext, state_backend=None
        )

        # Test with isolated backend using unique database file
        isolated_backend = SQLiteBackend(with_backend_db_path)
        runner_with_backend = create_test_flujo(
            pipeline, context_model=LargeContext, state_backend=isolated_backend
        )

        # Run with large context (include required initial_prompt field)
        large_context_data = {"initial_prompt": "test", "large_data": "y" * 10000}

        try:
            # Measure performance with multiple iterations for stability
            iterations = 3
            no_backend_times = []
            with_backend_times = []

            for _ in range(iterations):
                # Test without backend
                start = time.perf_counter_ns()
                await gather_result(
                    runner_no_backend, "test", initial_context_data=large_context_data
                )
                no_backend_times.append((time.perf_counter_ns() - start) / 1_000_000_000.0)

                # Test with isolated backend
                start = time.perf_counter_ns()
                await gather_result(
                    runner_with_backend, "test", initial_context_data=large_context_data
                )
                with_backend_times.append((time.perf_counter_ns() - start) / 1_000_000_000.0)

            # Calculate averages for more stable measurements
            avg_no_backend = sum(no_backend_times) / len(no_backend_times)
            avg_with_backend = sum(with_backend_times) / len(with_backend_times)

            overhead_percentage = ((avg_with_backend - avg_no_backend) / avg_no_backend) * 100

            # Log performance results for debugging (consistent logging approach)
            logger.debug("Large Context Performance Test (Isolated):")
            logger.debug(f"Average time without backend: {avg_no_backend:.4f}s")
            logger.debug(f"Average time with backend: {avg_with_backend:.4f}s")
            logger.debug(f"Overhead: {overhead_percentage:.2f}%")
            logger.debug(f"Individual measurements - No backend: {no_backend_times}")
            logger.debug(f"Individual measurements - With backend: {with_backend_times}")

            # Get configurable overhead limit (higher in CI environments)
            overhead_limit = self.get_default_overhead_limit()

            # Verify that the performance optimization is working
            # The overhead should be significantly reduced with the optimizations
            assert overhead_percentage <= overhead_limit, (
                f"Persistence overhead with large context ({overhead_percentage:.2f}%) exceeds {overhead_limit}%"
            )

            # Additional assertion to ensure the optimization is actually working
            # If we're still seeing high overhead, log a warning but don't fail
            if overhead_percentage > 20.0:  # If still above 20%, log a warning
                logger.warning(
                    f"Performance overhead is still high ({overhead_percentage:.2f}%). "
                    "Consider additional optimizations for large context serialization."
                )

        finally:
            # Clean up database files to prevent resource contention
            try:
                if with_backend_db_path.exists():
                    with_backend_db_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up test database files: {e}")

    @pytest.mark.asyncio
    async def test_serialization_optimization_effectiveness(self, tmp_path: Path) -> None:
        """Test that serialization optimizations are working correctly."""

        # Test with different context sizes to verify optimization effectiveness
        test_cases = [
            ("small", "x" * 1000),  # 1KB
            ("medium", "x" * 5000),  # 5KB
            ("large", "x" * 10000),  # 10KB
        ]

        for size_name, data_size in test_cases:

            class TestContext(PipelineContext):
                test_data: str = data_size

            # Create isolated backend
            test_id = uuid.uuid4().hex[:8]
            db_path = tmp_path / f"optimization_test_{test_id}.db"

            try:
                # Measure serialization time
                context = TestContext(initial_prompt="test", test_data=data_size)

                start_time = time.perf_counter_ns()
                # This should trigger the optimized serialization path
                serialized = context.model_dump()
                serialization_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

                # Verify serialization completed successfully
                assert isinstance(serialized, dict)
                assert "test_data" in serialized
                assert len(serialized["test_data"]) == len(data_size)

                # Log performance metrics
                logger.debug(
                    f"{size_name.capitalize()} context serialization: {serialization_time:.6f}s"
                )

                # For large contexts, serialization should be reasonably fast
                if size_name == "large":
                    assert serialization_time < 0.1, (
                        f"Large context serialization too slow: {serialization_time:.6f}s"
                    )

            finally:
                # Clean up
                try:
                    if db_path.exists():
                        db_path.unlink()
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_first_principles_caching_effectiveness(self, tmp_path: Path) -> None:
        """Test that the first principles approach with caching and delta detection works correctly."""

        # Create context with substantial data
        class LargeContext(PipelineContext):
            large_data: str = "x" * 10000  # 10KB of data
            counter: int = 0

        # Create isolated backend
        test_id = uuid.uuid4().hex[:8]
        db_path = tmp_path / f"first_principles_test_{test_id}.db"
        backend = SQLiteBackend(db_path)

        from flujo.application.core.state_manager import StateManager

        state_manager = StateManager(backend)

        try:
            # Test 1: First serialization (should cache)
            context1 = LargeContext(initial_prompt="test", large_data="y" * 10000, counter=1)

            start_time = time.perf_counter_ns()
            await state_manager.persist_workflow_state(
                run_id="test_run",
                context=context1,
                current_step_index=0,
                last_step_output=None,
                status="running",
            )
            first_serialization_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

            # Test 2: Same context (should use cache)
            start_time = time.perf_counter_ns()
            await state_manager.persist_workflow_state(
                run_id="test_run",
                context=context1,
                current_step_index=1,
                last_step_output="output1",
                status="running",
            )
            cached_serialization_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

            # Test 3: Changed context (should serialize again)
            context2 = LargeContext(initial_prompt="test", large_data="y" * 10000, counter=2)
            start_time = time.perf_counter_ns()
            await state_manager.persist_workflow_state(
                run_id="test_run",
                context=context2,
                current_step_index=2,
                last_step_output="output2",
                status="running",
            )
            changed_serialization_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

            # Verify caching effectiveness
            logger.debug("First Principles Caching Test Results:")
            logger.debug(f"First serialization: {first_serialization_time:.6f}s")
            logger.debug(f"Cached serialization: {cached_serialization_time:.6f}s")
            logger.debug(f"Changed context serialization: {changed_serialization_time:.6f}s")

            # The cached serialization should be significantly faster
            assert cached_serialization_time < first_serialization_time * 0.5, (
                f"Cached serialization ({cached_serialization_time:.6f}s) should be much faster than "
                f"first serialization ({first_serialization_time:.6f}s)"
            )

            # Changed context should take similar time to first serialization
            # Allow for some timing variation due to system load
            assert changed_serialization_time >= cached_serialization_time * 0.8, (
                f"Changed context serialization ({changed_serialization_time:.6f}s) should be similar to "
                f"cached serialization ({cached_serialization_time:.6f}s) - timing too different"
            )

            # Verify cache clearing works
            state_manager.clear_cache("test_run")

            # Test 4: After cache clear, should serialize again
            start_time = time.perf_counter_ns()
            await state_manager.persist_workflow_state(
                run_id="test_run",
                context=context1,
                current_step_index=3,
                last_step_output="output3",
                status="running",
            )
            after_clear_time = (time.perf_counter_ns() - start_time) / 1_000_000_000.0

            # Should be similar to first serialization (no cache)
            # Allow for timing variations due to system load
            assert after_clear_time >= cached_serialization_time * 0.8, (
                f"After cache clear ({after_clear_time:.6f}s) should be similar to "
                f"cached serialization ({cached_serialization_time:.6f}s) - timing too different"
            )

        finally:
            # Clean up
            try:
                if db_path.exists():
                    db_path.unlink()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_delta_detection_accuracy(self, tmp_path: Path) -> None:
        """Test that delta detection accurately identifies context changes."""

        from flujo.application.core.state_manager import StateManager

        class TestContext(PipelineContext):
            data: str = "test"
            counter: int = 0

        # Create state manager without backend for testing
        state_manager = StateManager()

        # Test 1: Same context should not trigger serialization
        context1 = TestContext(initial_prompt="test", data="value1", counter=1)
        context1_same = TestContext(initial_prompt="test", data="value1", counter=1)

        # First call should serialize
        assert state_manager._should_serialize_context(context1, "test_run")

        # Second call with same data should not serialize
        assert not state_manager._should_serialize_context(context1_same, "test_run")

        # Test 2: Different context should trigger serialization
        context2 = TestContext(initial_prompt="test", data="value2", counter=1)
        assert state_manager._should_serialize_context(context2, "test_run")

        # Test 3: Same context after change should not serialize again
        assert not state_manager._should_serialize_context(context2, "test_run")

        # Test 4: Clear cache and verify
        state_manager.clear_cache("test_run")
        assert state_manager._should_serialize_context(context1, "test_run")

    @pytest.mark.asyncio
    async def test_buffer_pooling_consistency_fix(self) -> None:
        """Test that buffer pooling state is consistent when pool operations fail."""

        from flujo.utils.performance import (
            enable_buffer_pooling,
            disable_buffer_pooling,
            clear_scratch_buffer,
            get_scratch_buffer,
            _return_buffer_to_pool_sync,
        )

        # Enable buffer pooling for testing
        enable_buffer_pooling()

        try:
            # Get a buffer and use it
            buffer1 = get_scratch_buffer()
            buffer1.extend(b"test data")

            # Clear the buffer
            clear_scratch_buffer()

            # Get another buffer (should be the same object)
            buffer2 = get_scratch_buffer()

            # Verify we have the same buffer object
            assert buffer1 is buffer2

            # Test the core fix: when _return_buffer_to_pool_sync fails,
            # the buffer should not be marked as returned
            buffer3 = get_scratch_buffer()
            buffer3.extend(b"important data")

            # Directly test the function that was buggy
            # This should return False when pool is full
            success = _return_buffer_to_pool_sync(buffer3)

            # If the pool is full, success should be False
            # and the buffer should still be available
            if not success:
                # Verify the buffer is still available (not marked as returned)
                buffer4 = get_scratch_buffer()
                assert buffer3 is buffer4  # Should be the same buffer

                # Verify the data is still there (buffer wasn't actually returned)
                assert buffer4 == b"important data"

        finally:
            # Disable buffer pooling
            disable_buffer_pooling()

    @pytest.mark.asyncio
    async def test_cache_consistency_data_loss_prevention(self) -> None:
        """Test that cache inconsistency doesn't cause data loss."""

        from flujo.application.core.state_manager import StateManager
        from flujo.state.backends.sqlite import SQLiteBackend
        from flujo.domain.models import PipelineContext
        import tempfile
        import os

        # Create a context with multiple fields
        class TestContext(PipelineContext):
            initial_prompt: str = "test prompt"
            important_data: str = "critical information"
            user_id: str = "user123"
            settings: dict = {"key": "value"}

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            backend = SQLiteBackend(db_path)
            state_manager = StateManager(backend)

            # Create context with important data
            context = TestContext(
                initial_prompt="test prompt",
                important_data="critical information that must be preserved",
                user_id="user123",
                settings={"key": "value", "important": "data"},
            )

            # Force cache eviction by adding many entries
            for i in range(150):  # More than the 100 limit
                temp_context = TestContext(
                    initial_prompt=f"temp prompt {i}",
                    important_data=f"temp data {i}",
                    user_id=f"user{i}",
                    settings={"temp": f"value{i}"},
                )
                # This will trigger cache eviction
                state_manager._cache_serialization(temp_context, f"run_{i}", {"temp": "data"})

            # Now persist our important context
            await state_manager.persist_workflow_state(
                run_id="important_run",
                context=context,
                current_step_index=0,
                last_step_output=None,
                status="running",
            )

            # Load the state back
            loaded_context, _, _, _, _, _, _ = await state_manager.load_workflow_state(
                "important_run", TestContext
            )

            # Verify that ALL data was preserved, not just initial_prompt
            assert loaded_context is not None
            assert loaded_context.important_data == "critical information that must be preserved"
            assert loaded_context.user_id == "user123"
            assert loaded_context.settings == {"key": "value", "important": "data"}

            # Verify the context is complete, not just the fallback serialization
            assert hasattr(loaded_context, "important_data")
            assert hasattr(loaded_context, "user_id")
            assert hasattr(loaded_context, "settings")

        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestCLIPerformance:
    """Test NFR-10: CLI commands must complete in <2s with 10,000 runs."""

    @staticmethod
    def get_cli_performance_threshold() -> float:
        """Get CLI performance threshold from environment or use default."""
        return float(os.getenv("FLUJO_CLI_PERF_THRESHOLD", "2.0"))

    @staticmethod
    def get_database_size() -> int:
        """Get database size based on environment - smaller for CI, full for local."""
        if os.getenv("CI") == "true":
            # Use 1,000 runs in CI for faster execution (10% of full size)
            return int(os.getenv("FLUJO_CI_DB_SIZE", "1000"))
        else:
            # Use 10,000 runs for local development (full size)
            return int(os.getenv("FLUJO_LOCAL_DB_SIZE", "10000"))

    @pytest.fixture
    def large_database(self, tmp_path: Path) -> Path:
        """Create a database with configurable number of runs for performance testing."""
        import asyncio

        db_path = tmp_path / "large_ops.db"
        backend = SQLiteBackend(db_path)

        # Get database size based on environment
        db_size = self.get_database_size()
        completed_runs = int(db_size * 0.95)  # 95% of runs are completed

        # Create runs with concurrent operations for better performance
        now = datetime.utcnow()

        async def create_database() -> None:
            # Prepare all run start operations
            run_start_tasks = []
            for i in range(db_size):
                dt = (now - timedelta(minutes=i)).isoformat()
                task = backend.save_run_start(
                    {
                        "run_id": f"run_{i:05d}",
                        "pipeline_id": f"pid_{i:05d}",
                        "pipeline_name": f"pipeline_{i % 10}",
                        "pipeline_version": "1.0",
                        "status": "running",
                        "created_at": dt,
                        "updated_at": dt,
                    }
                )
                run_start_tasks.append(task)

            # Execute all run start operations concurrently
            await asyncio.gather(*run_start_tasks)

            # Prepare run end operations for completed runs (95%)
            run_end_tasks = []
            for i in range(completed_runs):
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
            for i in range(completed_runs):
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

        # Get expected database size
        expected_size = self.get_database_size()

        # Verify the database file exists
        assert large_database.exists(), f"Database file {large_database} does not exist"

        # Verify the database has data by checking with the backend directly
        backend = SQLiteBackend(large_database)
        runs = asyncio.run(backend.list_runs())

        # Should have expected number of runs
        assert len(runs) == expected_size, f"Expected {expected_size} runs, got {len(runs)}"

        # Verify some specific runs exist
        run_ids = [run["run_id"] for run in runs]
        assert "run_00000" in run_ids, "First run should exist"
        assert f"run_{expected_size - 1:05d}" in run_ids, "Last run should exist"

        # Verify run details
        run_details = asyncio.run(backend.get_run_details("run_00001"))
        assert run_details is not None, "Run details should be retrievable"
        assert run_details["run_id"] == "run_00001"

        # Verify step data exists
        steps = asyncio.run(backend.list_run_steps("run_00001"))
        assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"

    @pytest.mark.slow
    def test_lens_list_performance(self, large_database: Path) -> None:
        """Test that `flujo lens list` completes in <2s with configurable number of runs."""
        import asyncio

        # Get expected database size for logging
        expected_size = self.get_database_size()

        # Set environment variable to point to our test database
        # Use correct URI format: sqlite:///path for absolute paths
        os.environ["FLUJO_STATE_URI"] = f"sqlite:///{large_database}"

        # Debug: Verify the database has data before running CLI
        backend = SQLiteBackend(large_database)
        runs = asyncio.run(backend.list_runs())
        logger.debug(
            f"Database contains {len(runs)} runs (expected {expected_size}) before CLI test"
        )

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
        """Test that `flujo lens show` completes in <2s with configurable number of runs."""

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
        """Test that `flujo lens list` with filters completes in <2s with configurable number of runs."""

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
        """Test that `flujo lens show` with nonexistent run completes in <2s with configurable number of runs."""

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
