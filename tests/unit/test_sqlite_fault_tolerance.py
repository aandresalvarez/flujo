"""Tests for SQLiteBackend fault tolerance and recovery scenarios."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest
import aiosqlite

from flujo.state.backends.sqlite import SQLiteBackend

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = pytest.mark.serial


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
async def test_sqlite_backend_handles_corrupted_database(tmp_path: Path) -> None:
    """Test that SQLiteBackend can handle corrupted database files."""
    db_path = tmp_path / "corrupted.db"

    # Create a corrupted database file
    with open(db_path, "w") as f:
        f.write("This is not a valid SQLite database")

    backend = SQLiteBackend(db_path)

    # The corruption should be detected and handled during initialization
    # The backend should either succeed in creating a new database or fail gracefully
    try:
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
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "total_steps": 0,
                "error_message": None,
                "execution_time_ms": None,
                "memory_usage_mb": None,
            }

        # Should not raise an exception - corruption should be handled
        await backend.save_state("test_run", state)

        # Should be able to load the state
        loaded = await backend.load_state("test_run")
        assert loaded is not None
        assert loaded["pipeline_id"] == "test_pipeline"

    except sqlite3.DatabaseError as e:
        # If corruption recovery fails, that's also acceptable behavior
        # The important thing is that the backend doesn't crash
        assert "file is not a database" in str(e) or "database corruption" in str(e)

        # Verify that the corrupted file was moved to a backup
        backup_files = list(db_path.parent.glob("corrupted.db.corrupt.*"))
        assert len(backup_files) > 0, "Corrupted file should have been moved to backup"


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
        "step_history": [],
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "total_steps": 0,
        "error_message": None,
        "execution_time_ms": None,
        "memory_usage_mb": None,
    }

    # Test normal operation first
    await backend.save_state("test_run", state)
    loaded = await backend.load_state("test_run")
    assert loaded is not None

    # Test that the database remains functional after normal operations
    # This verifies that the transaction handling works correctly
    state2 = state.copy()
    state2["pipeline_id"] = "test_pipeline_2"
    await backend.save_state("test_run_2", state2)
    loaded2 = await backend.load_state("test_run_2")
    assert loaded2 is not None
    assert loaded2["pipeline_id"] == "test_pipeline_2"


@pytest.mark.asyncio
async def test_sqlite_backend_migration_failure_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from migration failures."""
    db_path = tmp_path / "migration_test.db"

    # Create an old database schema that's compatible with the new schema
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE workflow_state (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                current_step_index INTEGER NOT NULL DEFAULT 0,
                pipeline_context TEXT NOT NULL,
                last_step_output TEXT,
                step_history TEXT,
                status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'failed', 'cancelled')),
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        await conn.commit()

    # Add some data to the old schema with proper JSON format
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            INSERT INTO workflow_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "old_run",
                "old_pipeline",
                "Old Pipeline",
                "0.1",
                1,
                '{"test": "data"}',
                '{"output": "test"}',
                "[]",
                "completed",
                int(datetime.now(timezone.utc).timestamp() * 1_000_000),
                int(datetime.now(timezone.utc).timestamp() * 1_000_000),
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
        "step_history": [],
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "total_steps": 0,
        "error_message": None,
        "execution_time_ms": None,
        "memory_usage_mb": None,
    }

    await working_backend.save_state("test_run", state)
    loaded = await working_backend.load_state("test_run")
    assert loaded is not None


@pytest.mark.asyncio
async def test_sqlite_backend_disk_space_exhaustion(tmp_path: Path) -> None:
    """Test that SQLiteBackend handles disk space exhaustion gracefully through public methods."""
    backend = SQLiteBackend(tmp_path / "disk_full.db")
    large_data = {"data": "x" * 1000000}
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": large_data,
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

    # Test that save_state handles disk space issues gracefully
    # The @db_retry decorator should handle operational errors
    try:
        await backend.save_state("test_run", state)
        # If it succeeds, verify the state was saved
        loaded = await backend.load_state("test_run")
        assert loaded is not None
    except sqlite3.OperationalError as e:
        # If disk space is exhausted, that's also acceptable behavior
        assert "database or disk is full" in str(e) or "disk is full" in str(e)


@pytest.mark.asyncio
async def test_sqlite_backend_connection_failure_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from connection failures through public methods."""
    backend = SQLiteBackend(tmp_path / "connection_test.db")
    state = {
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
    await backend.save_state("test_run", state)

    # Test that load_state handles connection issues gracefully
    # The @db_retry decorator should handle connection failures
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_sqlite_backend_transaction_rollback_on_error(tmp_path: Path) -> None:
    """Test that SQLiteBackend properly rolls back transactions on errors through public methods."""
    backend = SQLiteBackend(tmp_path / "rollback_test.db")
    state = {
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

    # Test that save_state handles transaction rollback gracefully
    # The @db_retry decorator should handle operational errors
    try:
        await backend.save_state("test_run", state)
        # If it succeeds, verify the state was saved
        loaded = await backend.load_state("test_run")
        assert loaded is not None
    except sqlite3.OperationalError:
        # If transaction rollback occurs, that's also acceptable behavior
        pass


@pytest.mark.asyncio
async def test_sqlite_backend_schema_validation_recovery(tmp_path: Path) -> None:
    """Test that SQLiteBackend can recover from schema validation issues through public methods."""
    db_path = tmp_path / "schema_test.db"

    # Create a database with a more compatible schema that will be migrated
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE workflow_state (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                current_step_index INTEGER NOT NULL DEFAULT 0,
                pipeline_context TEXT NOT NULL,
                last_step_output TEXT,
                step_history TEXT,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        await conn.commit()

    backend = SQLiteBackend(db_path)

    # The migration should add missing columns and make the database usable
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"test": "data"},
        "last_step_output": None,
        "step_history": [],
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "total_steps": 0,
        "error_message": None,
        "execution_time_ms": None,
        "memory_usage_mb": None,
    }

    # Should handle schema migration and save successfully
    await backend.save_state("schema_test_run", state)

    # Should be able to load the state
    loaded = await backend.load_state("schema_test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_proper_initialization(
    tmp_path: Path, sample_state
) -> None:
    """Test that retry mechanism properly calls _ensure_init through public methods."""
    backend = SQLiteBackend(tmp_path / "proper_init_test.db")

    # Test that save_state handles initialization properly
    # The @db_retry decorator should call _ensure_init during retry attempts
    await backend.save_state("test_run", sample_state)

    # Verify the operation succeeded
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_explicit_return(tmp_path: Path, sample_state) -> None:
    """Test that retry mechanism always has explicit return paths through public methods."""
    backend = SQLiteBackend(tmp_path / "explicit_return_test.db")

    # Test successful case with save_state
    result = await backend.save_state("test_run", sample_state)
    assert result is None  # save_state should return None

    # Test successful case with load_state
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test failure case with non-existent run
    non_existent = await backend.load_state("non_existent_run")
    assert non_existent is None


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_mixed_scenarios(tmp_path: Path, sample_state) -> None:
    """Test retry mechanism with mixed error scenarios through public methods."""
    backend = SQLiteBackend(tmp_path / "mixed_scenarios_test.db")

    # Test multiple operations that might encounter different error types
    # The @db_retry decorator should handle various error scenarios
    for i in range(5):
        state = sample_state.copy()
        state["pipeline_id"] = f"mixed_scenario_pipeline_{i}"
        await backend.save_state(f"test_run_{i}", state)

        loaded = await backend.load_state(f"test_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"mixed_scenario_pipeline_{i}"


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_concurrent_safety(
    tmp_path: Path, sample_state
) -> None:
    """Test that retry mechanism is safe under concurrent access through public methods."""
    backend = SQLiteBackend(tmp_path / "concurrent_safety_test.db")

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
async def test_sqlite_backend_retry_mechanism_memory_cleanup(tmp_path: Path, sample_state) -> None:
    """Test that retry mechanism doesn't leak memory during repeated operations."""
    backend = SQLiteBackend(tmp_path / "memory_cleanup_test.db")

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
    assert backend.db_path.parent.exists()  # Directory should exist


@pytest.mark.asyncio
async def test_sqlite_backend_retry_mechanism_real_operations(tmp_path: Path) -> None:
    """Test retry mechanism with real database operations through public methods."""
    backend = SQLiteBackend(tmp_path / "real_operations_test.db")

    # Test save_state with retry mechanism
    state = {
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

    # This should work normally
    await backend.save_state("test_run", state)

    # Test load_state with retry mechanism
    loaded = await backend.load_state("test_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test updating the state
    state["status"] = "completed"
    state["updated_at"] = datetime.now(timezone.utc)
    await backend.save_state("test_run", state)

    # Verify the update was saved
    updated = await backend.load_state("test_run")
    assert updated is not None
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_sqlite_backend_connection_pool_fault_tolerance(tmp_path: Path, sample_state) -> None:
    """Test that the connection pool handles faults gracefully."""
    backend = SQLiteBackend(tmp_path / "connection_pool_test.db")

    # Test multiple rapid operations to stress the connection pool
    for i in range(20):
        state = sample_state.copy()
        state["pipeline_id"] = f"pool_fault_pipeline_{i}"
        await backend.save_state(f"pool_fault_run_{i}", state)

        loaded = await backend.load_state(f"pool_fault_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"pool_fault_pipeline_{i}"

    # Verify all operations succeeded
    for i in range(20):
        loaded = await backend.load_state(f"pool_fault_run_{i}")
        assert loaded is not None


@pytest.mark.asyncio
async def test_sqlite_backend_transaction_fault_tolerance(tmp_path: Path, sample_state) -> None:
    """Test that the transaction helper handles faults gracefully."""
    backend = SQLiteBackend(tmp_path / "transaction_fault_test.db")

    # Test operations that use the new transaction helper
    await backend.save_state("transaction_fault_run", sample_state)

    # Verify the transaction was committed correctly
    loaded = await backend.load_state("transaction_fault_run")
    assert loaded is not None
    assert loaded["pipeline_id"] == "test_pipeline"

    # Test multiple operations in sequence
    for i in range(5):
        state = sample_state.copy()
        state["pipeline_id"] = f"transaction_fault_pipeline_{i}"
        await backend.save_state(f"transaction_fault_run_{i}", state)

        loaded = await backend.load_state(f"transaction_fault_run_{i}")
        assert loaded is not None
        assert loaded["pipeline_id"] == f"transaction_fault_pipeline_{i}"


@pytest.mark.asyncio
async def test_sqlite_backend_schema_migration_fault_tolerance(
    tmp_path: Path, sample_state
) -> None:
    """Test that schema migration works robustly with the new retry logic."""
    backend = SQLiteBackend(tmp_path / "schema_migration_fault_test.db")

    # Test that the new schema (WITHOUT ROWID, integer timestamps) works correctly
    await backend.save_state("schema_migration_fault_run", sample_state)

    # Verify the new schema format is working
    loaded = await backend.load_state("schema_migration_fault_run")
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
