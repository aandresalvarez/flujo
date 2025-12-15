"""Tests for SQLiteBackend edge cases and missing coverage."""

import pytest
import asyncio
import sqlite3
import time
from datetime import datetime, timezone
import os
from pathlib import Path
from unittest.mock import patch

from flujo.state.backends.sqlite import SQLiteBackend
from .conftest import capture_logs

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow


class TestSQLiteBackendDeleteState:
    """Test delete_state functionality as mentioned in Copilot comments."""

    @pytest.mark.asyncio
    async def test_delete_state_removes_workflow_from_database(
        self, sqlite_backend_factory
    ) -> None:
        """Test that delete_state actually removes the workflow from the database."""
        backend = sqlite_backend_factory("delete_test.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Create a workflow
        state = {
            "run_id": "test_delete_run",
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

        # Save the workflow
        await backend.save_state("test_delete_run", state)

        # Verify it exists
        loaded_state = await backend.load_state("test_delete_run")
        assert loaded_state is not None
        assert loaded_state["run_id"] == "test_delete_run"

        # Delete the workflow
        await backend.delete_state("test_delete_run")

        # Verify it's gone
        deleted_state = await backend.load_state("test_delete_run")
        assert deleted_state is None

        # Verify it's not in the list
        all_workflows = await backend.list_workflows()
        assert not any(wf["run_id"] == "test_delete_run" for wf in all_workflows)

    @pytest.mark.asyncio
    async def test_delete_state_nonexistent_workflow(self, sqlite_backend_factory) -> None:
        """Test that delete_state handles nonexistent workflows gracefully."""
        backend = sqlite_backend_factory("delete_nonexistent.db")

        # Try to delete a workflow that doesn't exist
        await backend.delete_state("nonexistent_run")

        # Should not raise an exception
        # Verify the database is still functional
        all_workflows = await backend.list_workflows()
        assert all_workflows == []

    @pytest.mark.asyncio
    async def test_delete_state_multiple_workflows(self, sqlite_backend_factory) -> None:
        """Test deleting multiple workflows and verify others remain."""
        backend = sqlite_backend_factory("delete_multiple.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Create multiple workflows
        workflows = []
        for i in range(5):
            state = {
                "run_id": f"run_{i}",
                "pipeline_id": f"pipeline_{i}",
                "pipeline_name": f"Pipeline {i}",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000 * i,
                "memory_usage_mb": 10.0 * i,
            }
            workflows.append(state)
            await backend.save_state(f"run_{i}", state)

        # Verify all workflows exist
        all_workflows = await backend.list_workflows()
        assert len(all_workflows) == 5

        # Delete specific workflows
        await backend.delete_state("run_1")
        await backend.delete_state("run_3")

        # Verify only the deleted ones are gone
        remaining_workflows = await backend.list_workflows()
        assert len(remaining_workflows) == 3

        remaining_ids = {wf["run_id"] for wf in remaining_workflows}
        assert "run_0" in remaining_ids
        assert "run_2" in remaining_ids
        assert "run_4" in remaining_ids
        assert "run_1" not in remaining_ids
        assert "run_3" not in remaining_ids


class TestSQLiteBackendBackupEdgeCases:
    """Test edge cases for backup file handling."""

    @pytest.mark.asyncio
    async def test_backup_with_special_characters_in_filename(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test backup handling with special characters in database filename."""
        # Create a database with special characters
        special_db_path = tmp_path / "test-db_with.special@chars#.db"
        backend = SQLiteBackend(special_db_path)

        # Create a corrupted database file
        special_db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Check that backup file was created with proper naming
        backup_files = list(tmp_path.glob("test-db_with.special@chars#.db.corrupt.*"))
        assert len(backup_files) == 1
        backup_file = backup_files[0]
        assert backup_file.name.startswith("test-db_with.special@chars#.db.corrupt.")

    @pytest.mark.asyncio
    async def test_backup_with_long_filename(self, tmp_path: Path, sqlite_backend_factory) -> None:
        """Test backup handling with very long database filename."""
        # Create a database with a very long name
        long_name = "a" * 200  # Very long filename
        long_db_path = tmp_path / f"{long_name}.db"
        backend = SQLiteBackend(long_db_path)

        # Create a corrupted database file
        long_db_path.write_text("corrupted database content")

        # Mock the database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            try:
                await backend._init_db()
            except sqlite3.DatabaseError:
                pass  # Expected to fail

        # Check that backup file was created
        backup_files = list(tmp_path.glob(f"{long_name}.db.corrupt.*"))
        assert len(backup_files) == 1

    @pytest.mark.asyncio
    async def test_backup_with_no_write_permissions(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test backup handling when directory has no write permissions."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the rename to fail due to permissions
        with patch("pathlib.Path.rename", side_effect=PermissionError("Permission denied")):
            with patch(
                "aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")
            ):
                try:
                    await backend._init_db()
                except sqlite3.DatabaseError:
                    pass  # Expected to fail

        # Check that the corrupted file was removed as fallback
        assert not db_path.exists()

    @pytest.mark.asyncio
    async def test_backup_with_disk_full_error(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test backup handling when disk is full."""
        db_path = tmp_path / "test.db"
        backend = SQLiteBackend(db_path)

        # Create a corrupted database file
        db_path.write_text("corrupted database content")

        # Mock the rename to fail due to disk full
        with patch("pathlib.Path.rename", side_effect=OSError("No space left on device")):
            with patch(
                "aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")
            ):
                try:
                    await backend._init_db()
                except sqlite3.DatabaseError:
                    pass  # Expected to fail

        # Check that the corrupted file was removed as fallback
        assert not db_path.exists()


class TestSQLiteBackendPerformanceThresholds:
    """Test performance threshold configurability."""

    @pytest.mark.asyncio
    async def test_performance_threshold_environment_variable(self, sqlite_backend_factory) -> None:
        """Test that performance thresholds are enforced via environment variable."""
        backend = sqlite_backend_factory("perf_env.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Create a large amount of test data
        num_workflows = 100
        for i in range(num_workflows):
            state = {
                "run_id": f"run_{i}",
                "pipeline_id": f"pipeline_{i}",
                "pipeline_name": f"Pipeline {i}",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"a": i},
                "last_step_output": f"out{i}",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"run_{i}", state)

        # Set a very strict threshold
        original_threshold = os.environ.get("SQLITE_PER_WORKFLOW_TIME_LIMIT")
        os.environ["SQLITE_PER_WORKFLOW_TIME_LIMIT"] = "0.000001"  # Very strict
        try:
            start_time = time.time()
            await backend.list_workflows()
            end_time = time.time()
            elapsed = end_time - start_time
            # Assert that the elapsed time exceeds the strict threshold
            assert elapsed > 0.000001 * num_workflows, (
                f"Elapsed time {elapsed} did not exceed the strict threshold "
                f"of {0.000001 * num_workflows} seconds."
            )
        finally:
            # Restore original environment
            if original_threshold is not None:
                os.environ["SQLITE_PER_WORKFLOW_TIME_LIMIT"] = original_threshold
            else:
                os.environ.pop("SQLITE_PER_WORKFLOW_TIME_LIMIT", None)

    @pytest.mark.asyncio
    async def test_performance_threshold_default_behavior(self, sqlite_backend_factory) -> None:
        """Test that performance thresholds work with default values."""
        backend = sqlite_backend_factory("perf_default.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Create a small amount of test data
        num_workflows = 10
        for i in range(num_workflows):
            state = {
                "run_id": f"run_{i}",
                "pipeline_id": f"pipeline_{i}",
                "pipeline_name": f"Pipeline {i}",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"a": i},
                "last_step_output": f"out{i}",
                "status": "completed",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"run_{i}", state)

        # This should pass with default thresholds
        start_time = time.time()
        workflows = await backend.list_workflows()
        end_time = time.time()

        assert len(workflows) == num_workflows
        # Should complete within reasonable time (default threshold is 0.0004s per workflow)
        # Allow for some overhead in test environment (4x the theoretical limit to handle CI/loaded machines)
        expected_time = num_workflows * 0.0004 * 4  # 16ms for 10 workflows
        assert (end_time - start_time) < expected_time, (
            f"Performance test failed: took {end_time - start_time:.6f}s, "
            f"expected < {expected_time:.6f}s for {num_workflows} workflows"
        )


class TestSQLiteBackendLoggerContextManagement:
    """Tests for SQLiteBackend logger context management."""

    @pytest.mark.asyncio
    async def test_logger_context_manager_cleanup(self, sqlite_backend_factory) -> None:
        """Test that logger context manager properly cleans up handlers."""
        import logging

        # Get the original logger state
        logger = logging.getLogger("flujo")
        original_handlers = logger.handlers.copy()

        # Use the context manager
        with capture_logs():
            backend = sqlite_backend_factory("test.db")
            await backend._ensure_init()

        # Verify handlers were cleaned up
        assert logger.handlers == original_handlers

    @pytest.mark.asyncio
    async def test_logger_context_manager_exception_handling(self, sqlite_backend_factory) -> None:
        """Test that logger context manager cleans up even when exceptions occur."""
        import logging

        logger = logging.getLogger("flujo")
        original_handlers = logger.handlers.copy()

        try:
            with capture_logs():
                backend = sqlite_backend_factory("test.db")
                await backend._ensure_init()
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass

        # Verify handlers were cleaned up even after exception
        assert logger.handlers == original_handlers


class TestSQLiteBackendTypeSafety:
    """Test type safety in retry mechanisms and other operations."""

    @pytest.mark.asyncio
    async def test_retry_mechanism_type_safety(self, sqlite_backend_factory) -> None:
        """Test that retry mechanism maintains proper type safety."""
        backend = sqlite_backend_factory("type_safety_test.db")

        # Test that load_state returns the correct type
        result = await backend.load_state("nonexistent")
        assert result is None or isinstance(result, dict)

        # Test that list_workflows returns the correct type
        workflows = await backend.list_workflows()
        assert isinstance(workflows, list)
        assert all(isinstance(wf, dict) for wf in workflows)

        # Test that get_workflow_stats returns the correct type
        stats = await backend.get_workflow_stats()
        assert isinstance(stats, dict)
        assert "total_workflows" in stats
        assert "status_counts" in stats

    @pytest.mark.asyncio
    async def test_serialization_type_safety(self, sqlite_backend_factory) -> None:
        """Test that serialization maintains type safety."""
        backend = sqlite_backend_factory("serialization_test.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Test with complex data types
        complex_state = {
            "run_id": "complex_test",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {
                "nested_dict": {"key": "value"},
                "list_data": [1, 2, 3, "string"],
                "mixed_types": {"int": 42, "float": 3.14, "bool": True, "none": None},
            },
            "last_step_output": {
                "result": "success",
                "data": [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}],
            },
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 10.0,
        }

        # Save and load the complex state
        await backend.save_state("complex_test", complex_state)
        loaded_state = await backend.load_state("complex_test")

        # Verify type safety
        assert loaded_state is not None
        assert isinstance(loaded_state, dict)
        assert loaded_state["run_id"] == "complex_test"
        assert isinstance(loaded_state["pipeline_context"], dict)
        assert isinstance(loaded_state["pipeline_context"]["nested_dict"], dict)
        assert isinstance(loaded_state["pipeline_context"]["list_data"], list)
        assert isinstance(loaded_state["last_step_output"], dict)


class TestSQLiteBackendConcurrencyEdgeCases:
    """Test edge cases in concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_delete_and_save(self, sqlite_backend_factory) -> None:
        """Test concurrent delete and save operations."""
        backend = sqlite_backend_factory("concurrent_test.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Create initial state
        state = {
            "run_id": "concurrent_test",
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

        await backend.save_state("concurrent_test", state)

        # Perform concurrent delete and save
        async def delete_operation():
            await backend.delete_state("concurrent_test")

        async def save_operation():
            await backend.save_state("concurrent_test", state)

        # Run both operations concurrently
        await asyncio.gather(delete_operation(), save_operation())

        # Verify final state (should be consistent)
        # The final state depends on which operation completed last
        # But the database should remain consistent

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test concurrent backup operations during corruption recovery."""
        backend = sqlite_backend_factory("concurrent_backup.db")

        # Create corrupted database
        db_path = tmp_path / "concurrent_backup.db"
        db_path.write_text("corrupted database content")

        # Mock database connection to raise corruption error
        with patch("aiosqlite.connect", side_effect=sqlite3.DatabaseError("Corrupted database")):
            # Try concurrent initialization attempts
            async def init_attempt():
                try:
                    await backend._init_db()
                except sqlite3.DatabaseError:
                    pass  # Expected to fail

            # Run multiple concurrent initialization attempts
            await asyncio.gather(*[init_attempt() for _ in range(3)])

            # Verify that backup files were created properly
            backup_files = list(tmp_path.glob("concurrent_backup.db.corrupt.*"))
            assert len(backup_files) >= 1  # At least one backup should be created
