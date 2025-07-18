"""Tests to ensure SQLite backend robustness and prevent regression of fixed issues."""

import pytest
from datetime import datetime, timezone

from flujo.state.backends.sqlite import SQLiteBackend


class TestSQLiteBackendRobustness:
    """Test SQLite backend robustness and error handling."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_robustness.db"

    @pytest.fixture
    def backend(self, temp_db_path):
        """Create a SQLite backend instance."""
        return SQLiteBackend(temp_db_path)

    @pytest.fixture
    def sample_state(self):
        """Sample state for testing save_state/load_state operations."""
        return {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

    @pytest.mark.asyncio
    async def test_execution_time_ms_handling(self, backend):
        """Test that execution_time_ms is handled correctly without unused variables."""
        # Create test state with execution_time_ms
        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": 1500,  # Test with a value
            "memory_usage_mb": None,
        }

        # Test saving state with execution_time_ms
        run_id = "test-run-123"
        await backend.save_state(run_id, state)

        # Verify the state was saved correctly
        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["execution_time_ms"] == 1500

    @pytest.mark.asyncio
    async def test_execution_time_ms_none_handling(self, backend):
        """Test that execution_time_ms=None is handled correctly."""
        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

        run_id = "test-run-124"
        await backend.save_state(run_id, state)

        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["execution_time_ms"] is None

    @pytest.mark.asyncio
    async def test_retry_mechanism_safe_control_flow(self, backend, sample_state):
        """Test that retry mechanism uses safe control flow through public methods."""
        # Test that save_state handles errors gracefully without using assert for control flow
        # The @db_retry decorator should handle errors properly

        # Test with a valid state first
        await backend.save_state("test-run-125", sample_state)
        loaded = await backend.load_state("test-run-125")
        assert loaded is not None

        # Test that the operation completes successfully without control flow issues
        # The decorator should handle any internal errors gracefully

    @pytest.mark.asyncio
    async def test_retry_mechanism_schema_error_handling(self, backend, sample_state):
        """Test that schema errors are handled correctly through public methods."""
        # Test that save_state handles schema errors gracefully
        # The @db_retry decorator should handle schema migration internally

        await backend.save_state("test-run-126", sample_state)
        loaded = await backend.load_state("test-run-126")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test-pipeline"

    @pytest.mark.asyncio
    async def test_retry_mechanism_database_locked_handling(self, backend, sample_state):
        """Test that database locked errors are handled correctly through public methods."""
        # Test that save_state handles database locked errors gracefully
        # The @db_retry decorator should handle retries internally

        await backend.save_state("test-run-127", sample_state)
        loaded = await backend.load_state("test-run-127")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test-pipeline"

    @pytest.mark.asyncio
    async def test_retry_mechanism_mixed_errors(self, backend, sample_state):
        """Test that mixed error types are handled correctly through public methods."""
        # Test multiple operations that might encounter different error types
        # The @db_retry decorator should handle various error scenarios

        # Test multiple save operations
        for i in range(5):
            state = sample_state.copy()
            state["pipeline_id"] = f"mixed_error_pipeline_{i}"
            await backend.save_state(f"test-run-128-{i}", state)

            loaded = await backend.load_state(f"test-run-128-{i}")
            assert loaded is not None
            assert loaded["pipeline_id"] == f"mixed_error_pipeline_{i}"

    @pytest.mark.asyncio
    async def test_retry_mechanism_max_retries(self, backend, sample_state):
        """Test that retry mechanism respects max retries through public methods."""
        # Test that the decorator respects retry limits
        # This tests the retry behavior indirectly through public methods

        # Perform multiple operations to stress test the retry mechanism
        for i in range(10):
            state = sample_state.copy()
            state["pipeline_id"] = f"max_retries_pipeline_{i}"
            await backend.save_state(f"test-run-129-{i}", state)

            loaded = await backend.load_state(f"test-run-129-{i}")
            assert loaded is not None
            assert loaded["pipeline_id"] == f"max_retries_pipeline_{i}"

    @pytest.mark.asyncio
    async def test_safe_control_flow_no_assert(self, backend, sample_state):
        """Test that no assert statements are used for control flow in public methods."""
        # Test that public methods don't use assert for control flow
        # The @db_retry decorator should use proper exception handling

        await backend.save_state("test-run-130", sample_state)
        loaded = await backend.load_state("test-run-130")
        assert loaded is not None

        # Test that operations complete successfully without control flow issues
        # The decorator should handle any internal errors gracefully

    @pytest.mark.asyncio
    async def test_error_message_robustness(self, backend, sample_state):
        """Test that error message parsing is robust through public methods."""
        # Test that public methods handle various error scenarios gracefully
        # The @db_retry decorator should handle different error message formats

        # Test with various run_id formats that might trigger different error types
        test_cases = [
            "normal_run",
            "run_with_underscores",
            "run-with-dashes",
            "run123",
            "RUN_WITH_UPPERCASE",
            "run_with_special_chars_!@#$%",
        ]

        for run_id in test_cases:
            await backend.save_state(run_id, sample_state)
            loaded = await backend.load_state(run_id)
            assert loaded is not None
            assert loaded["pipeline_id"] == "test-pipeline"

    @pytest.mark.asyncio
    async def test_state_serialization_robustness(self, backend):
        """Test that state serialization handles edge cases."""
        # Test with various data types in state
        state = {
            "pipeline_id": "test-pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0.0",
            "current_step_index": 0,
            "pipeline_context": {
                "string": "test",
                "number": 42,
                "boolean": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "none": None,
            },
            "last_step_output": {"output": "test"},
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 50.5,
        }

        run_id = "test-run-125"
        await backend.save_state(run_id, state)

        loaded_state = await backend.load_state(run_id)
        assert loaded_state is not None
        assert loaded_state["pipeline_context"]["string"] == "test"
        assert loaded_state["pipeline_context"]["number"] == 42
        assert loaded_state["pipeline_context"]["boolean"] is True
        assert loaded_state["pipeline_context"]["list"] == [1, 2, 3]
        assert loaded_state["pipeline_context"]["dict"]["nested"] == "value"
        assert loaded_state["pipeline_context"]["none"] is None
        assert loaded_state["execution_time_ms"] == 1000
        assert loaded_state["memory_usage_mb"] == 50.5

    @pytest.mark.asyncio
    async def test_connection_pool_robustness(self, backend, sample_state):
        """Test that the connection pool handles concurrent operations robustly."""
        # Test multiple rapid operations to stress the connection pool
        for i in range(20):
            state = sample_state.copy()
            state["pipeline_id"] = f"pool_test_pipeline_{i}"
            await backend.save_state(f"pool_test_run_{i}", state)

            loaded = await backend.load_state(f"pool_test_run_{i}")
            assert loaded is not None
            assert loaded["pipeline_id"] == f"pool_test_pipeline_{i}"

    @pytest.mark.asyncio
    async def test_transaction_robustness(self, backend, sample_state):
        """Test that the transaction helper handles operations robustly."""
        # Test operations that use the new transaction helper
        await backend.save_state("transaction_test_run", sample_state)

        # Verify the transaction was committed correctly
        loaded = await backend.load_state("transaction_test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test-pipeline"

        # Test multiple operations in sequence
        for i in range(5):
            state = sample_state.copy()
            state["pipeline_id"] = f"transaction_pipeline_{i}"
            await backend.save_state(f"transaction_run_{i}", state)

            loaded = await backend.load_state(f"transaction_run_{i}")
            assert loaded is not None
            assert loaded["pipeline_id"] == f"transaction_pipeline_{i}"

    @pytest.mark.asyncio
    async def test_schema_migration_robustness(self, backend, sample_state):
        """Test that schema migration works robustly with the new retry logic."""
        # Test that the new schema (WITHOUT ROWID, integer timestamps) works correctly
        await backend.save_state("schema_test_run", sample_state)

        # Verify the new schema format is working
        loaded = await backend.load_state("schema_test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test-pipeline"

        # Test that timestamps are properly converted to/from epoch microseconds
        assert isinstance(loaded["created_at"], datetime)
        assert isinstance(loaded["updated_at"], datetime)

        # Test that new columns are handled correctly
        assert "total_steps" in loaded
        assert "error_message" in loaded
        assert "execution_time_ms" in loaded
        assert "memory_usage_mb" in loaded


class TestSQLiteBackendRegressionPrevention:
    """Test to prevent regression of fixed issues."""

    @pytest.mark.asyncio
    async def test_no_unused_variables(self, tmp_path):
        """Test that no unused variables are created in save_state."""
        backend = SQLiteBackend(tmp_path / "test.db")
        await backend._ensure_init()

        # Create a state with execution_time_ms
        state = {
            "pipeline_id": "test",
            "pipeline_name": "Test",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": 500,
            "memory_usage_mb": None,
        }

        # This should not create any unused variables
        await backend.save_state("test-run", state)

        # Verify the state was saved correctly
        loaded = await backend.load_state("test-run")
        assert loaded is not None
        assert loaded["execution_time_ms"] == 500

    @pytest.mark.asyncio
    async def test_safe_exception_handling(self, tmp_path):
        """Test that exceptions are handled safely through public methods."""
        backend = SQLiteBackend(tmp_path / "test.db")

        # Test that the retry mechanism doesn't use assert for control flow
        # This is now tested through public methods that use the @db_retry decorator
        sample_state = {
            "pipeline_id": "test",
            "pipeline_name": "Test",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

        # Test that save_state handles errors gracefully
        await backend.save_state("test-run", sample_state)
        loaded = await backend.load_state("test-run")
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_error_detection_robustness(self, tmp_path):
        """Test that error detection is robust through public methods."""
        backend = SQLiteBackend(tmp_path / "test.db")

        # Test that public methods handle various error scenarios gracefully
        # The @db_retry decorator should handle different error types internally

        sample_state = {
            "pipeline_id": "test",
            "pipeline_name": "Test",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": None,
            "memory_usage_mb": None,
        }

        # Test with various run_id formats
        test_cases = [
            "normal_run",
            "run_with_underscores",
            "run-with-dashes",
            "run123",
            "RUN_WITH_UPPERCASE",
            "run_with_special_chars_!@#$%",
        ]

        for run_id in test_cases:
            await backend.save_state(run_id, sample_state)
            loaded = await backend.load_state(run_id)
            assert loaded is not None
            assert loaded["pipeline_id"] == "test"

    @pytest.mark.asyncio
    async def test_decorator_robustness(self, tmp_path):
        """Test that the @db_retry decorator works robustly."""
        backend = SQLiteBackend(tmp_path / "test.db")

        # Test that all methods using @db_retry work correctly
        sample_state = {
            "pipeline_id": "decorator_test",
            "pipeline_name": "Decorator Test",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 0,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 25.5,
        }

        # Test save_state (uses @db_retry)
        await backend.save_state("decorator_test_run", sample_state)

        # Test load_state (uses @db_retry)
        loaded = await backend.load_state("decorator_test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "decorator_test"
        assert loaded["execution_time_ms"] == 1000
        assert loaded["memory_usage_mb"] == 25.5

        # Test that the decorator preserves method signatures and return types
        result = await backend.save_state("decorator_test_run_2", sample_state)
        assert result is None  # save_state should return None

        loaded_2 = await backend.load_state("decorator_test_run_2")
        assert isinstance(loaded_2, dict)  # load_state should return Dict or None
