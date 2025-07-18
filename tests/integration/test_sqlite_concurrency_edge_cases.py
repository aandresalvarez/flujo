"""Integration tests for SQLiteBackend concurrency edge cases."""

import asyncio
import time
from datetime import datetime
from pathlib import Path

import pytest
import sqlite3

from flujo.state.backends.sqlite import SQLiteBackend

pytestmark = pytest.mark.serial


class TestSQLiteConcurrencyEdgeCases:
    """Tests for SQLiteBackend concurrency edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(self, tmp_path: Path) -> None:
        """Test concurrent backup operations."""

        async def create_backup(i: int) -> None:
            db_path = tmp_path / f"test{i}.db"
            db_path.write_bytes(b"corrupted sqlite data")
            backend = SQLiteBackend(db_path)

            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state(f"test_run_{i}", sample_state)

        # Run multiple concurrent backup operations
        tasks = [create_backup(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify backup files were created during the failed recovery attempts
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_concurrent_database_initialization(self, tmp_path: Path) -> None:
        """Test concurrent database initialization."""
        db_path = tmp_path / "concurrent_test.db"

        async def init_database():
            backend = SQLiteBackend(db_path)
            # Trigger initialization through a public method
            await backend.save_state(
                "test_run",
                {
                    "pipeline_id": "test_pipeline",
                    "pipeline_name": "Test Pipeline",
                    "pipeline_version": "1.0",
                    "current_step_index": 0,
                    "pipeline_context": {"test": "data"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "total_steps": 0,
                    "error_message": None,
                    "execution_time_ms": None,
                    "memory_usage_mb": None,
                },
            )
            return backend

        # Run multiple concurrent initialization attempts
        tasks = [init_database() for _ in range(10)]
        backends = await asyncio.gather(*tasks)

        # Verify all backends are initialized
        for backend in backends:
            assert backend._initialized

        # Verify database file exists and is valid
        assert db_path.exists()
        assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_concurrent_save_operations(self, tmp_path: Path) -> None:
        """Test concurrent save operations."""
        backend = SQLiteBackend(tmp_path / "concurrent_save.db")

        async def save_operation(i: int):
            state = {
                "run_id": f"concurrent_run_{i}",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "running",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"concurrent_run_{i}", state)

        # Run multiple concurrent save operations
        tasks = [save_operation(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all states were saved
        workflows = await backend.list_workflows()
        assert len(workflows) == 20

    @pytest.mark.asyncio
    async def test_concurrent_load_operations(self, tmp_path: Path) -> None:
        """Test concurrent load operations."""
        backend = SQLiteBackend(tmp_path / "concurrent_load.db")

        # Create test data
        for i in range(10):
            state = {
                "run_id": f"load_test_{i}",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "completed",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"load_test_{i}", state)

        async def load_operation(i: int):
            return await backend.load_state(f"load_test_{i}")

        # Run multiple concurrent load operations
        tasks = [load_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all loads were successful
        for i, result in enumerate(results):
            assert result is not None
            assert result["run_id"] == f"load_test_{i}"

    @pytest.mark.asyncio
    async def test_concurrent_delete_operations(self, tmp_path: Path) -> None:
        """Test concurrent delete operations."""
        backend = SQLiteBackend(tmp_path / "concurrent_delete.db")

        # Create test data
        for i in range(10):
            state = {
                "run_id": f"delete_test_{i}",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "completed",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"delete_test_{i}", state)

        async def delete_operation(i: int):
            await backend.delete_state(f"delete_test_{i}")

        # Run multiple concurrent delete operations
        tasks = [delete_operation(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all states were deleted
        workflows = await backend.list_workflows()
        assert len(workflows) == 0

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, tmp_path: Path) -> None:
        """Test concurrent mixed operations (save, load, delete)."""
        backend = SQLiteBackend(tmp_path / "concurrent_mixed.db")

        async def mixed_operation(i: int):
            # Save operation
            state = {
                "run_id": f"mixed_test_{i}",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "running",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"mixed_test_{i}", state)

            # Load operation
            loaded = await backend.load_state(f"mixed_test_{i}")
            assert loaded is not None

            # Delete operation (for even indices)
            if i % 2 == 0:
                await backend.delete_state(f"mixed_test_{i}")

        # Run multiple concurrent mixed operations
        tasks = [mixed_operation(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify final state
        workflows = await backend.list_workflows()
        assert len(workflows) == 10  # Only odd indices should remain

    @pytest.mark.asyncio
    async def test_concurrent_backup_with_corruption(self, tmp_path: Path) -> None:
        """Test concurrent backup operations with corruption."""

        async def create_corrupted_backup(i: int) -> None:
            db_path = tmp_path / f"corrupt_test{i}.db"
            db_path.write_bytes(b"corrupted sqlite data")
            backend = SQLiteBackend(db_path)

            sample_state = {
                "pipeline_id": f"test_pipeline_{i}",
                "pipeline_name": f"test_pipeline_{i}",
                "pipeline_version": "1.0.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": None,
                "step_history": [],
                "status": "running",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }

            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                await backend.save_state(f"test_run_{i}", sample_state)

        # Run multiple concurrent backup operations with corruption
        tasks = [create_corrupted_backup(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify backup files were created during the failed recovery attempts
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) > 0

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, tmp_path: Path) -> None:
        """Test concurrent error handling scenarios."""

        async def error_operation(i: int):
            db_path = tmp_path / f"error_test{i}.db"
            backend = SQLiteBackend(db_path)

            try:
                sample_state = {
                    "pipeline_id": f"test_pipeline_{i}",
                    "pipeline_name": f"test_pipeline_{i}",
                    "pipeline_version": "1.0.0",
                    "current_step_index": 0,
                    "pipeline_context": {"test": "data"},
                    "last_step_output": None,
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }

                await backend.save_state(f"test_run_{i}", sample_state)
                return True
            except Exception:
                return False

        # Run multiple concurrent operations with potential errors
        tasks = [error_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify most operations succeeded
        success_count = sum(results)
        assert success_count >= 8  # At least 80% should succeed

    @pytest.mark.asyncio
    async def test_concurrent_performance_under_load(self, tmp_path: Path) -> None:
        """Test performance under concurrent load."""

        async def performance_operation(i: int):
            # Create state
            state = {
                "pipeline_id": f"perf_pipeline_{i}",
                "pipeline_name": f"Performance Pipeline {i}",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i, "data": "x" * 1000},
                "last_step_output": f"output_{i}",
                "status": "running",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "total_steps": 10,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }

            backend = SQLiteBackend(tmp_path / f"perf_test{i}.db")
            await backend.save_state(f"perf_run_{i}", state)
            loaded = await backend.load_state(f"perf_run_{i}")
            assert loaded is not None
            await backend.delete_state(f"perf_run_{i}")

        # Run many concurrent operations to test performance
        start_time = time.time()
        tasks = [performance_operation(i) for i in range(50)]
        await asyncio.gather(*tasks)
        end_time = time.time()

        # Verify operations completed in reasonable time
        duration = end_time - start_time
        assert duration < 30  # Should complete within 30 seconds
