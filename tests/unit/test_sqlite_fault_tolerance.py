"""Tests for SQLiteBackend fault tolerance and recovery scenarios."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pytest
import aiosqlite

from flujo.state.backends.sqlite import SQLiteBackend

# Mark all tests in this module for serial execution to prevent SQLite concurrency issues
pytestmark = [pytest.mark.serial, pytest.mark.slow]


@pytest.fixture(autouse=True)
async def cleanup_sqlite_backends(monkeypatch):
    """Auto-cleanup all SQLiteBackend instances created in this module."""
    backends = []
    original_init = SQLiteBackend.__init__

    def tracking_init(self, *args, **kwargs):
        backends.append(self)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "__init__", tracking_init)
    yield

    # Cleanup all backends with timeout (fault tolerance tests may have corrupted backends)
    for backend in backends:
        try:
            await asyncio.wait_for(backend.close(), timeout=2.0)
        except (Exception, asyncio.TimeoutError):
            pass  # Best effort cleanup - corrupted backends may not close cleanly


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
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    await backend.save_state("test_run", state)

    # Prepare a real connection for the retry
    real_conn = await aiosqlite.connect(tmp_path / "connection_test.db")

    call_count = 0

    async def fake_connect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_conn

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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
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
    now = datetime.now(timezone.utc).replace(microsecond=0)
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
