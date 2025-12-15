"""Fuzzing tests for SQLiteBackend to catch edge cases and potential issues."""

import pytest
import asyncio
import sqlite3
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import time

from flujo.state.backends.sqlite import SQLiteBackend


def sanitize_filename(filename: str) -> str:
    safe = "".join(c for c in filename if c.isalnum() or c in "._-")
    if safe.endswith(".db"):
        safe = safe[:-3]
    return safe or "test"


class TestSQLiteBackendFuzzing:
    """Fuzzing tests to catch edge cases and potential issues."""

    # Mark all fuzzing tests as slow and stress since they are resource-intensive
    pytestmark = [pytest.mark.slow, pytest.mark.stress]

    @pytest.mark.asyncio
    async def test_fuzz_database_filename_edge_cases(self, tmp_path: Path) -> None:
        """Test database initialization with various filename edge cases."""
        edge_case_filenames = [
            "",  # Empty filename
            "a" * 1000,  # Very long filename
            "test.db" * 50,  # Repetitive filename
            "test.db" + "\x00" + "test.db",  # Null bytes
            "test.db" + "\n" + "test.db",  # Newlines
            "test.db" + "\t" + "test.db",  # Tabs
            "test.db" + " " * 100 + "test.db",  # Many spaces
            "test.db" + "\\" + "test.db",  # Backslashes
            "test.db" + "/" + "test.db",  # Forward slashes
            "test.db" + ":" + "test.db",  # Colons (Windows)
            "test.db" + "*" + "test.db",  # Wildcards
            "test.db" + "?" + "test.db",  # Question marks
            "test.db" + "<" + "test.db",  # Less than
            "test.db" + ">" + "test.db",  # Greater than
            "test.db" + "|" + "test.db",  # Pipe
            "test.db" + '"' + "test.db",  # Quotes
        ]

        for filename in edge_case_filenames:
            try:
                # Clean the filename to make it filesystem-safe
                safe_filename = sanitize_filename(filename)

                db_path = tmp_path / f"{safe_filename}.db"
                backend = SQLiteBackend(db_path)

                # Try to initialize
                await backend._ensure_init()

                # Verify it works
                assert backend._initialized

            except Exception as e:
                # Some edge cases might fail, but they shouldn't crash
                assert "database" in str(e).lower() or "file" in str(e).lower()

    @pytest.mark.asyncio
    async def test_fuzz_state_data_edge_cases(self, tmp_path: Path) -> None:
        """Test saving and loading state with various data edge cases."""
        backend = SQLiteBackend(tmp_path / "fuzz_data.db")

        edge_case_data = [
            # Empty data
            {},
            # Very large data
            {"pipeline_context": {"large_data": "x" * 1000000}},
            # Unicode data
            {"pipeline_context": {"unicode": "🎉🚀💻"}},
            # Special characters
            {"pipeline_context": {"special": "!@#$%^&*()_+-=[]{}|;':\",./<>?"}},
            # Nested structures
            {"pipeline_context": {"nested": {"deep": {"structure": {"value": 42}}}}},
            # Mixed types
            {"pipeline_context": {"mixed": [1, "string", 3.14, True, None, {"key": "value"}]}},
            # Binary-like data
            {"pipeline_context": {"binary": bytes(range(256)).decode("latin1")}},
            # Very long keys
            {"pipeline_context": {"a" * 1000: "value"}},
            # Very long values
            {"pipeline_context": {"key": "a" * 10000}},
        ]

        for i, edge_data in enumerate(edge_case_data):
            try:
                now = datetime.now(timezone.utc).replace(microsecond=0)
                state = {
                    "run_id": f"fuzz_test_{i}",
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "Test Pipeline",
                    "pipeline_version": "1.0",
                    "current_step_index": 0,
                    "pipeline_context": edge_data.get("pipeline_context", {}),
                    "last_step_output": edge_data.get("last_step_output", None),
                    "status": "running",
                    "created_at": now,
                    "updated_at": now,
                    "total_steps": 5,
                    "error_message": None,
                    "execution_time_ms": 1000,
                    "memory_usage_mb": 10.0,
                }

                # Save the state
                await backend.save_state(f"fuzz_test_{i}", state)

                # Load the state
                loaded_state = await backend.load_state(f"fuzz_test_{i}")

                # Verify it loaded correctly
                assert loaded_state is not None
                assert loaded_state["run_id"] == f"fuzz_test_{i}"

            except Exception as e:
                # Some edge cases might fail due to size limits, but shouldn't crash
                assert (
                    "size" in str(e).lower()
                    or "memory" in str(e).lower()
                    or "limit" in str(e).lower()
                )

    @pytest.mark.asyncio
    async def test_fuzz_concurrent_operations(self, tmp_path: Path) -> None:
        """Test concurrent operations with random timing."""
        backend = SQLiteBackend(tmp_path / "fuzz_concurrent.db")

        async def random_operation(operation_id: int):
            """Perform a random operation with random delays."""
            await asyncio.sleep(random.uniform(0, 0.1))

            operation = random.choice(["save", "load", "delete", "list"])

            if operation == "save":
                now = datetime.now(timezone.utc).replace(microsecond=0)
                state = {
                    "run_id": f"fuzz_run_{operation_id}",
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "Test Pipeline",
                    "pipeline_version": "1.0",
                    "current_step_index": random.randint(0, 10),
                    "pipeline_context": {"random": random.randint(1, 1000)},
                    "last_step_output": f"output_{operation_id}",
                    "status": random.choice(["running", "completed", "failed"]),
                    "created_at": now,
                    "updated_at": now,
                    "total_steps": random.randint(1, 20),
                    "error_message": None,
                    "execution_time_ms": random.randint(100, 10000),
                    "memory_usage_mb": random.uniform(1.0, 100.0),
                }
                await backend.save_state(f"fuzz_run_{operation_id}", state)

            elif operation == "load":
                await backend.load_state(f"fuzz_run_{operation_id}")

            elif operation == "delete":
                await backend.delete_state(f"fuzz_run_{operation_id}")

            elif operation == "list":
                await backend.list_workflows(limit=random.randint(1, 50))

        # Run many concurrent operations
        operations = [random_operation(i) for i in range(50)]
        await asyncio.gather(*operations)

        # Verify database is still functional
        workflows = await backend.list_workflows()
        assert isinstance(workflows, list)

    @pytest.mark.asyncio
    async def test_fuzz_backup_recovery_edge_cases(self, tmp_path: Path) -> None:
        """Test backup recovery with various corruption scenarios."""
        edge_case_corruptions = [
            # Empty file
            "",
            # Very large file
            "x" * 1000000,
            # Partial SQLite header
            "SQLite format 3",
            # Corrupted SQLite header
            "SQLite format 3\x00\x00\x00\x00\x00\x00\x00\x00",
            # Random binary data
            bytes(random.getrandbits(8) for _ in range(1000)).decode("latin1"),
            # Unicode corruption
            "🎉🚀💻" * 1000,
            # Null bytes
            "\x00" * 1000,
            # Control characters
            "".join(chr(i) for i in range(32)) * 100,
            # Very long lines
            "a" * 10000 + "\n" + "b" * 10000,
        ]

        for i, corruption in enumerate(edge_case_corruptions):
            try:
                db_path = tmp_path / f"fuzz_corrupt_{i}.db"
                backend = SQLiteBackend(db_path)

                # Create corrupted database
                db_path.write_text(corruption)

                # Mock database connection to raise corruption error
                with patch(
                    "aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")
                ):
                    try:
                        await backend._init_db()
                    except sqlite3.DatabaseError:
                        pass  # Expected to fail

                # Verify backup was created
                backup_files = list(tmp_path.glob(f"fuzz_corrupt_{i}.db.corrupt.*"))
                assert len(backup_files) >= 1

            except Exception as e:
                # Some edge cases might fail, but shouldn't crash
                assert "file" in str(e).lower() or "permission" in str(e).lower()

    @pytest.mark.asyncio
    async def test_fuzz_schema_migration_edge_cases(self, tmp_path: Path) -> None:
        """Test schema migration with various edge cases."""
        backend = SQLiteBackend(tmp_path / "fuzz_schema.db")

        # Create a database with some data
        now = datetime.now(timezone.utc).replace(microsecond=0)
        state = {
            "run_id": "fuzz_schema_test",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": None,
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 10.0,
        }

        await backend.save_state("fuzz_schema_test", state)

        # Test various schema migration scenarios
        migration_scenarios = [
            # Add new columns
            "ALTER TABLE workflow_state ADD COLUMN fuzz_column TEXT",
            # Drop columns (should fail)
            "ALTER TABLE workflow_state DROP COLUMN run_id",
            # Modify constraints
            "ALTER TABLE workflow_state MODIFY COLUMN status TEXT",
            # Create indexes
            "CREATE INDEX IF NOT EXISTS fuzz_index ON workflow_state(run_id)",
            # Drop indexes
            "DROP INDEX IF EXISTS fuzz_index",
        ]

        for scenario in migration_scenarios:
            try:
                # Try to execute the migration scenario
                import aiosqlite

                async with aiosqlite.connect(backend.db_path) as db:
                    await db.execute(scenario)
                    await db.commit()
            except Exception as e:
                # Some scenarios should fail, but shouldn't crash
                err = str(e).lower()
                assert (
                    "syntax" in err
                    or "constraint" in err
                    or "not supported" in err
                    or "cannot drop primary key" in err
                )

    @pytest.mark.asyncio
    async def test_fuzz_performance_under_load(self, tmp_path: Path) -> None:
        """Test performance under various load conditions."""
        backend = SQLiteBackend(tmp_path / "fuzz_performance.db")

        # Create many workflows
        num_workflows = 1000
        now = datetime.now(timezone.utc).replace(microsecond=0)

        for i in range(num_workflows):
            state = {
                "run_id": f"perf_test_{i}",
                "pipeline_id": f"pipeline_{i % 10}",
                "pipeline_name": f"Pipeline {i % 10}",
                "pipeline_version": "1.0",
                "current_step_index": i % 5,
                "pipeline_context": {"data": f"value_{i}"},
                "last_step_output": f"output_{i}",
                "status": random.choice(["running", "completed", "failed"]),
                "created_at": now - timedelta(minutes=i),
                "updated_at": now - timedelta(minutes=i),
                "total_steps": random.randint(1, 20),
                "error_message": None,
                "execution_time_ms": random.randint(100, 10000),
                "memory_usage_mb": random.uniform(1.0, 100.0),
            }
            await backend.save_state(f"perf_test_{i}", state)

        # Test various query patterns
        query_patterns = [
            lambda: backend.list_workflows(),
            lambda: backend.list_workflows(status="running"),
            lambda: backend.list_workflows(status="completed"),
            lambda: backend.list_workflows(pipeline_id="pipeline_1"),
            lambda: backend.get_workflow_stats(),
            lambda: backend.get_failed_workflows(hours_back=24),
            lambda: backend.cleanup_old_workflows(days_old=1),
        ]

        for pattern in query_patterns:
            start_time = time.time()
            result = await pattern()
            end_time = time.time()

            # Verify query completed within reasonable time
            assert (end_time - start_time) < 5.0  # 5 seconds max
            assert result is not None

    @pytest.mark.asyncio
    async def test_fuzz_error_recovery_scenarios(self, tmp_path: Path) -> None:
        """Test error recovery with various error scenarios."""
        error_scenarios = [
            # Database locked
            sqlite3.OperationalError("database is locked"),
            # Disk full
            sqlite3.OperationalError("disk I/O error"),
            # Permission denied
            sqlite3.OperationalError("unable to open database file"),
            # Corrupted database
            sqlite3.DatabaseError("database disk image is malformed"),
            # Schema mismatch
            sqlite3.OperationalError("no such column: nonexistent_column"),
            # Constraint violation
            sqlite3.IntegrityError("UNIQUE constraint failed"),
            # Timeout
            sqlite3.OperationalError("database is locked"),
            # Memory error
            sqlite3.OperationalError("out of memory"),
        ]

        for i, error in enumerate(error_scenarios):
            try:
                db_path = tmp_path / f"fuzz_error_{i}.db"
                backend = SQLiteBackend(db_path)

                # Mock database operations to raise the error
                with patch("aiosqlite.connect", side_effect=error):
                    try:
                        await backend._ensure_init()
                    except Exception as e:
                        # Should handle the error gracefully
                        assert isinstance(
                            e,
                            (
                                sqlite3.OperationalError,
                                sqlite3.DatabaseError,
                                sqlite3.IntegrityError,
                            ),
                        )

            except Exception as e:
                # Should not crash, but might raise expected errors
                assert isinstance(
                    e, (sqlite3.OperationalError, sqlite3.DatabaseError, sqlite3.IntegrityError)
                )
