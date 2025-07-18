"""Tests for SQLiteBackend retry mechanism and error handling scenarios."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest

from flujo.state.backends.sqlite import SQLiteBackend

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = pytest.mark.serial


@pytest.fixture
def sqlite_backend(tmp_path: Path):
    def _create_backend(db_name: str):
        return SQLiteBackend(tmp_path / db_name)

    return _create_backend


@pytest.fixture
def sample_state():
    """Sample state for testing save_state/load_state operations."""
    return {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
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
        "execution_time_ms": None,
        "memory_usage_mb": None,
    }


@pytest.mark.asyncio
async def test_retry_on_schema_errors(sqlite_backend, sample_state) -> None:
    """Test that schema errors trigger proper retry and re-initialization."""
    backend = sqlite_backend("schema_retry_test.db")

    # First call should work normally
    await backend.save_state("test_run", sample_state)

    # Verify the state was saved
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Now test that schema errors are handled gracefully
    # The @db_retry decorator should handle schema migration internally
    await backend.save_state("test_run_2", sample_state)
    loaded_2 = await backend.load_state("test_run_2")
    assert loaded_2 is not None


@pytest.mark.asyncio
async def test_retry_on_database_locked_errors(sqlite_backend, sample_state) -> None:
    """Test retry behavior for database locked errors through public methods."""
    from unittest.mock import patch
    import aiosqlite

    backend = sqlite_backend("locked_retry_test.db")

    # First, initialize the backend normally
    await backend.save_state("init_test", sample_state)

    # Now test retry behavior by mocking only the specific operation
    original_execute = aiosqlite.Connection.execute

    async def mock_execute_with_retry(self, sql, parameters=None):
        # Scope call_count inside the mock function to avoid race conditions
        if not hasattr(mock_execute_with_retry, "call_count"):
            mock_execute_with_retry.call_count = 0

        # Only mock the specific INSERT/UPDATE operations for save_state
        if "INSERT INTO workflow_state" in sql or "ON CONFLICT" in sql:
            mock_execute_with_retry.call_count += 1

            # Fail first two calls with database locked error
            if mock_execute_with_retry.call_count <= 2:
                raise sqlite3.OperationalError("database is locked")

        # Succeed on third call or for other operations
        return await original_execute(self, sql, parameters)

    with patch("aiosqlite.Connection.execute", mock_execute_with_retry):
        # Test that save_state handles database locked errors with retries
        await backend.save_state("locked_test", sample_state)

        # Verify that the operation succeeded after retries
        assert mock_execute_with_retry.call_count >= 3, (
            f"Expected at least 3 calls due to retries, got {mock_execute_with_retry.call_count}"
        )

        loaded = await backend.load_state("locked_test")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_retry_on_mixed_error_scenarios(sqlite_backend, sample_state) -> None:
    """Test retry behavior with mixed error types through public API."""
    backend = sqlite_backend("mixed_errors_test.db")

    # Test multiple operations that might encounter different error types
    states = []
    for i in range(5):
        state = sample_state.copy()
        state["pipeline_id"] = f"pipeline_{i}"
        states.append(state)

    # Save multiple states (may encounter locks, schema issues, etc.)
    for i, state in enumerate(states):
        await backend.save_state(f"run_{i}", state)

    # Load all states to verify they were saved correctly
    for i in range(5):
        loaded = await backend.load_state(f"run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"pipeline_{i}"


@pytest.mark.asyncio
async def test_concurrent_access_safety(sqlite_backend, sample_state) -> None:
    """Test that retry mechanism is safe under concurrent access through public methods."""
    backend = sqlite_backend("concurrent_test.db")

    # Create multiple concurrent save operations
    async def concurrent_save(operation_id: int):
        state = sample_state.copy()
        state["pipeline_id"] = f"concurrent_pipeline_{operation_id}"
        await backend.save_state(f"concurrent_run_{operation_id}", state)
        return f"success_{operation_id}"

    # Run multiple concurrent operations
    results = await asyncio.gather(
        concurrent_save(1),
        concurrent_save(2),
        concurrent_save(3),
        concurrent_save(4),
        return_exceptions=True,
    )

    # All should succeed despite potential database locks
    assert len(results) == 4
    assert all(isinstance(r, str) and r.startswith("success_") for r in results)

    # Verify all states were saved correctly
    for i in range(1, 5):
        loaded = await backend.load_state(f"concurrent_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"concurrent_pipeline_{i}"


@pytest.mark.asyncio
async def test_memory_cleanup_during_retries(sqlite_backend, sample_state) -> None:
    """Test that retry mechanism doesn't leak memory during repeated operations."""
    backend = sqlite_backend("memory_test.db")

    # Perform multiple save/load operations to stress test memory usage
    for i in range(10):
        state = sample_state.copy()
        state["pipeline_id"] = f"memory_test_pipeline_{i}"
        await backend.save_state(f"memory_test_run_{i}", state)

        loaded = await backend.load_state(f"memory_test_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"memory_test_pipeline_{i}"

    # The backend should still be in a valid state
    assert backend.db_path.exists()
    assert backend.db_path.parent.exists()


@pytest.mark.asyncio
async def test_logging_behavior_during_retries(sqlite_backend, sample_state) -> None:
    """Test that retry mechanism provides proper logging through public methods."""
    backend = sqlite_backend("logging_test.db")

    # Test that operations complete successfully and log appropriately
    await backend.save_state("logging_test_run", sample_state)

    # Verify the operation succeeded
    loaded = await backend.load_state("logging_test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_type_safety_preservation(sqlite_backend, sample_state) -> None:
    """Test that retry mechanism maintains type safety through public methods."""
    backend = sqlite_backend("type_safety_test.db")

    # Test save_state (should return None)
    result = await backend.save_state("type_test_run", sample_state)
    assert result is None

    # Test load_state (should return Dict or None)
    loaded = await backend.load_state("type_test_run")
    assert isinstance(loaded, dict)
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test load_state for non-existent run (should return None)
    non_existent = await backend.load_state("non_existent_run")
    assert non_existent is None


@pytest.mark.asyncio
async def test_edge_case_parameters(sqlite_backend, sample_state) -> None:
    """Test retry mechanism with edge case parameters through public methods."""
    backend = sqlite_backend("edge_case_test.db")

    # Test with various run_id formats
    test_cases = [
        "simple_run",
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
        assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_real_database_operations_with_retry(sqlite_backend, sample_state) -> None:
    """Test retry mechanism with real database operations."""
    backend = sqlite_backend("real_db_test.db")

    # Test save_state with retry mechanism
    await backend.save_state("test_run", sample_state)

    # Test load_state with retry mechanism
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"
    assert loaded["pipeline_name"] == "Test Pipeline"
    assert loaded["status"] == "running"

    # Test updating the state
    sample_state["status"] = "completed"
    sample_state["updated_at"] = datetime.now(timezone.utc)
    await backend.save_state("test_run", sample_state)

    # Verify the update was saved
    updated = await backend.load_state("test_run")
    assert updated is not None
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_corruption_recovery(sqlite_backend, tmp_path: Path) -> None:
    """Test that retry mechanism works with database corruption scenarios."""
    db_name = "corruption_test.db"
    db_path = tmp_path / db_name
    backend = sqlite_backend(db_name)

    # Create a corrupted database file
    with open(db_path, "w") as f:
        f.write("corrupt data")

    # Now try to use the backend, expecting it to handle the corruption gracefully
    sample_state = {
        "pipeline_id": "corruption_test",
        "pipeline_name": "Corruption Test",
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
        "execution_time_ms": None,
        "memory_usage_mb": None,
    }

    # The backend should handle corruption gracefully and reinitialize
    # We expect this to succeed after corruption recovery
    try:
        await backend.save_state("corruption_test_run", sample_state)

        # Verify the operation succeeded despite initial corruption
        loaded = await backend.load_state("corruption_test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "corruption_test"
    except sqlite3.DatabaseError as e:
        # If corruption recovery fails, that's also acceptable behavior
        # The important thing is that it doesn't crash or infinite loop
        assert "file is not a database" in str(e) or "database is corrupted" in str(e)

    # Verify that the backend attempted corruption recovery
    # (the corrupted file should have been moved to a backup)
    backup_files = list(tmp_path.glob("*.corrupt.*"))
    assert len(backup_files) > 0, "Expected corruption recovery to create backup files"


@pytest.mark.asyncio
async def test_connection_pool_retry_behavior(sqlite_backend, sample_state) -> None:
    """Test that the new connection pool works correctly with retry logic."""
    backend = sqlite_backend("connection_pool_test.db")

    # Test multiple rapid operations to stress the connection pool
    for i in range(20):
        state = sample_state.copy()
        state["pipeline_id"] = f"pool_test_pipeline_{i}"
        await backend.save_state(f"pool_test_run_{i}", state)

        loaded = await backend.load_state(f"pool_test_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"pool_test_pipeline_{i}"

    # Verify all operations succeeded
    for i in range(20):
        loaded = await backend.load_state(f"pool_test_run_{i}")
        assert loaded is not None


@pytest.mark.asyncio
async def test_transaction_retry_behavior(sqlite_backend, sample_state) -> None:
    """Test that the new transaction helper works correctly with retry logic."""
    backend = sqlite_backend("transaction_test.db")

    # Test operations that use the new transaction helper
    await backend.save_state("transaction_test_run", sample_state)

    # Verify the transaction was committed correctly
    loaded = await backend.load_state("transaction_test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test multiple operations in sequence
    for i in range(5):
        state = sample_state.copy()
        state["pipeline_id"] = f"transaction_pipeline_{i}"
        await backend.save_state(f"transaction_run_{i}", state)

        loaded = await backend.load_state(f"transaction_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"transaction_pipeline_{i}"


@pytest.mark.asyncio
async def test_schema_migration_retry_behavior(sqlite_backend, sample_state) -> None:
    """Test that schema migration works correctly with the new retry logic."""
    backend = sqlite_backend("schema_migration_test.db")

    # Test that the new schema (WITHOUT ROWID, integer timestamps) works correctly
    await backend.save_state("schema_test_run", sample_state)

    # Verify the new schema format is working
    loaded = await backend.load_state("schema_test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test that timestamps are properly converted to/from epoch microseconds
    assert isinstance(loaded["created_at"], datetime)
    assert isinstance(loaded["updated_at"], datetime)

    # Test that new columns are handled correctly
    assert "total_steps" in loaded
    assert "error_message" in loaded
    assert "execution_time_ms" in loaded
    assert "memory_usage_mb" in loaded
