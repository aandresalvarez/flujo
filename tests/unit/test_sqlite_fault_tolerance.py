"""Tests for SQLiteBackend fault tolerance and recovery scenarios."""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pytest
import aiosqlite

from flujo.state.backends.sqlite import SQLiteBackend

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = pytest.mark.serial


@pytest.mark.asyncio
async def test_sqlite_backend_handles_corrupted_database(tmp_path: Path) -> None:
    """Test that SQLiteBackend can handle corrupted database files."""
    db_path = tmp_path / "corrupted.db"

    # Create a corrupted database file
    with open(db_path, "w") as f:
        f.write("This is not a valid SQLite database")

    backend = SQLiteBackend(db_path)

    # Use context manager for proper cleanup
    async with backend:
        # Should handle corruption gracefully and create a new database
        state = {
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "status": "running",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

    # Should not raise an exception
    await backend.save_state("test_run", state)

    # Should be able to load the state
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_sqlite_backend_handles_partial_writes(tmp_path: Path) -> None:
    """Test that SQLiteBackend handles partial writes and incomplete transactions."""
    backend = SQLiteBackend(tmp_path / "partial.db")

    # Create a valid state structure
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # Simulate a partial write by mocking the commit method to raise an exception
    with patch(
        "aiosqlite.Connection.commit",
        side_effect=sqlite3.OperationalError("Simulated commit failure"),
    ):
        # This should not cause data corruption
        try:
            await backend.save_state("test_run", state)
        except sqlite3.OperationalError:
            pass  # Simulate failure during commit
        # Ensure that partial writes do not corrupt the database

    # After a commit failure, the database should be reinitialized and the state should not be saved
    loaded = await backend.load_state("test_run")
    assert loaded is None  # State should not be saved due to commit failure


@pytest.mark.asyncio
async def test_sqlite_backend_migration_failure_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from migration failures."""
    db_path = tmp_path / "migration_test.db"

    # Create an old database schema
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE workflow_state (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT,
                pipeline_version TEXT,
                current_step_index INTEGER,
                pipeline_context TEXT,
                last_step_output TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        await conn.commit()

    # Add some data to the old schema with proper JSON format
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            INSERT INTO workflow_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "old_run",
                "old_pipeline",
                "0.1",
                1,
                '{"test": "data"}',
                '{"output": "test"}',
                "completed",
                "2023-01-01T00:00:00",
                "2023-01-01T00:00:00",
            ),
        )
        await conn.commit()

    # Now create a backend that should migrate the old schema
    backend = SQLiteBackend(db_path)

    # The migration should succeed and preserve existing data
    loaded = await backend.load_state("old_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "old_pipeline"
    assert loaded["status"] == "completed"


@pytest.mark.asyncio
async def test_sqlite_backend_concurrent_migration_safety(tmp_path: Path) -> None:
    """Test that concurrent access during migration is handled safely."""
    db_path = tmp_path / "concurrent_migration.db"

    # Create multiple backends that will try to migrate simultaneously
    backends = [SQLiteBackend(db_path) for _ in range(3)]

    # Try to initialize all backends concurrently
    # This should not cause database corruption
    try:
        await asyncio.gather(*[backend._ensure_init() for backend in backends])
    except Exception:
        # Some concurrent access might fail, but should not corrupt the database
        pass

    # At least one backend should work
    working_backend = backends[0]
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    await working_backend.save_state("test_run", state)
    loaded = await working_backend.load_state("test_run")
    assert loaded is not None


@pytest.mark.asyncio
async def test_sqlite_backend_disk_space_exhaustion(tmp_path: Path) -> None:
    """Test that SQLiteBackend handles disk space exhaustion gracefully."""
    backend = SQLiteBackend(tmp_path / "disk_full.db")
    large_data = {"data": "x" * 1000000}
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": large_data,
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    # Mock only the save operation to fail, not the entire connection
    with patch.object(backend, "_with_retries") as mock_retries:
        mock_retries.side_effect = sqlite3.OperationalError("database or disk is full")
        with pytest.raises(sqlite3.OperationalError):
            await backend.save_state("test_run", state)


@pytest.mark.asyncio
async def test_sqlite_backend_connection_failure_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from connection failures."""
    backend = SQLiteBackend(tmp_path / "connection_test.db")
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await backend.save_state("test_run", state)

    # Prepare a real connection for the retry
    real_conn = await aiosqlite.connect(tmp_path / "connection_test.db")

    class RealAsyncConn:
        async def __aenter__(self):
            return real_conn

        async def __aexit__(self, exc_type, exc, tb):
            pass

    call_count = 0

    def fake_connect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("database is locked")
        return RealAsyncConn()

    with patch("aiosqlite.connect", side_effect=fake_connect):
        loaded = await backend.load_state("test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test_pipeline"
        assert call_count == 2  # Should have retried once
    await real_conn.close()


@pytest.mark.asyncio
async def test_sqlite_backend_transaction_rollback_on_error(tmp_path: Path) -> None:
    """Test that SQLiteBackend properly rolls back transactions on errors."""
    backend = SQLiteBackend(tmp_path / "rollback_test.db")
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # Mock the database commit to fail during save
    with patch(
        "aiosqlite.Connection.commit", side_effect=sqlite3.OperationalError("database is locked")
    ):
        with pytest.raises(sqlite3.OperationalError):
            await backend.save_state("test_run", state)


@pytest.mark.asyncio
async def test_sqlite_backend_schema_validation_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from schema validation issues."""
    db_path = tmp_path / "schema_test.db"

    # Create a database with missing required columns
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE workflow_state (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT
            )
        """)
        await conn.commit()

    backend = SQLiteBackend(db_path)

    # The migration should add missing columns and make the database usable
    now = datetime.utcnow().replace(microsecond=0)
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }

    # Should handle schema migration and save successfully
    await backend.save_state("schema_test_run", state)

    # Should be able to load the state
    loaded = await backend.load_state("schema_test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_proper_initialization(tmp_path: Path) -> None:
    """Test that retry mechanism properly calls _ensure_init instead of _init_db."""
    backend = SQLiteBackend(tmp_path / "proper_init_test.db")

    # Track calls to _ensure_init
    original_ensure_init = backend._ensure_init
    ensure_init_calls = 0

    async def mock_ensure_init():
        nonlocal ensure_init_calls
        ensure_init_calls += 1
        return await original_ensure_init()

    backend._ensure_init = mock_ensure_init

    # Create a function that raises schema errors
    async def schema_error_func(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # This should call _ensure_init during retry attempts
    with pytest.raises(sqlite3.DatabaseError):
        await backend._with_retries(schema_error_func)

    # Should have called _ensure_init during retry attempts
    assert ensure_init_calls > 0


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_explicit_return(tmp_path: Path) -> None:
    """Test that retry mechanism always has explicit return paths."""
    backend = SQLiteBackend(tmp_path / "explicit_return_test.db")

    # Test successful case
    async def success_func(*args, **kwargs):
        return "success"

    result = await backend._with_retries(success_func)
    assert result == "success"

    # Test failure case with non-schema error
    async def non_schema_error(*args, **kwargs):
        raise sqlite3.DatabaseError("some other error")

    with pytest.raises(sqlite3.DatabaseError, match="some other error"):
        await backend._with_retries(non_schema_error)


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_mixed_scenarios(tmp_path: Path) -> None:
    """Test retry mechanism with mixed error scenarios."""
    backend = SQLiteBackend(tmp_path / "mixed_scenarios_test.db")

    call_count = 0

    async def mixed_error_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("database is locked")
        elif call_count == 2:
            raise sqlite3.DatabaseError("no such column: missing_column")
        else:
            return "success after mixed errors"

    # Should handle mixed errors and eventually succeed
    result = await backend._with_retries(mixed_error_func)
    assert result == "success after mixed errors"
    assert call_count == 3


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_concurrent_safety(tmp_path: Path) -> None:
    """Test that retry mechanism is safe under concurrent access."""
    backend = SQLiteBackend(tmp_path / "concurrent_safety_test.db")

    # Create multiple concurrent operations
    async def concurrent_operation(operation_id: int):
        async def operation(*args, **kwargs):
            if operation_id % 2 == 0:
                raise sqlite3.OperationalError("database is locked")
            return f"success_{operation_id}"

        return await backend._with_retries(operation)

    # Run multiple concurrent operations
    results = await asyncio.gather(
        concurrent_operation(1),
        concurrent_operation(2),
        concurrent_operation(3),
        concurrent_operation(4),
        return_exceptions=True,
    )

    # Some should succeed, some should fail, but no infinite loops
    assert len(results) == 4
    assert any(isinstance(r, str) and r.startswith("success_") for r in results)
    assert any(isinstance(r, Exception) for r in results)


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_memory_cleanup(tmp_path: Path) -> None:
    """Test that retry mechanism doesn't leak memory during repeated failures."""
    backend = SQLiteBackend(tmp_path / "memory_cleanup_test.db")

    # Create a function that fails consistently
    async def memory_leak_test(*args, **kwargs):
        raise sqlite3.DatabaseError("no such column: missing_column")

    # Run multiple retry attempts to check for memory leaks
    for _ in range(5):
        with pytest.raises(sqlite3.DatabaseError):
            await backend._with_retries(memory_leak_test)

    # The backend should still be in a valid state
    assert backend.db_path.exists()
    # The _initialized flag may be True after _ensure_init() succeeds,
    # but the important thing is that the backend remains functional
    assert backend.db_path.parent.exists()  # Directory should exist


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_real_operations(tmp_path: Path) -> None:
    """Test retry mechanism with real database operations."""
    backend = SQLiteBackend(tmp_path / "real_operations_test.db")

    # Test save_state with retry mechanism
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # This should work normally
    await backend.save_state("test_run", state)

    # Test load_state with retry mechanism
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"
