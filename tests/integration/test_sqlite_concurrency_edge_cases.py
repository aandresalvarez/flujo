"""Integration tests for SQLiteBackend concurrency edge cases."""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flujo.state.backends.sqlite import SQLiteBackend
from flujo.type_definitions.common import JSONObject

pytestmark = [pytest.mark.serial, pytest.mark.slow]


class TestSQLiteConcurrencyEdgeCases:
    """Tests for SQLiteBackend concurrency edge cases."""

    @pytest.mark.asyncio
    async def test_sqlite_backend_multithreaded_access(self, tmp_path: Path) -> None:
        """Ensure shared backend works across distinct event loops/threads."""

        backend = SQLiteBackend(tmp_path / "shared.db")
        await backend._ensure_init()

        errors: list[BaseException] = []

        def run_in_thread(run_id: str, idx: int) -> JSONObject | None:
            async def _worker() -> JSONObject:
                state: JSONObject = {
                    "run_id": run_id,
                    "pipeline_id": "pipe",
                    "pipeline_name": "Pipe",
                    "pipeline_version": "v1",
                    "current_step_index": idx,
                    "pipeline_context": {"idx": idx},
                    "last_step_output": f"out-{idx}",
                    "step_history": [],
                    "status": "running",
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "total_steps": 1,
                    "error_message": None,
                    "execution_time_ms": 0,
                    "memory_usage_mb": 0.0,
                }
                await backend.save_state(run_id, state)
                loaded = await backend.load_state(run_id)
                assert loaded is not None
                return loaded

            try:
                return asyncio.run(_worker())
            except BaseException as exc:
                errors.append(exc)
                return None

        try:
            thread_tasks = [
                asyncio.to_thread(run_in_thread, "run-a", 1),
                asyncio.to_thread(run_in_thread, "run-b", 2),
            ]
            results = await asyncio.gather(*thread_tasks)
        finally:
            await backend.shutdown()

        assert not errors, f"Unexpected errors during multithreaded access: {errors}"
        run_ids = {result["run_id"] for result in results if result is not None}
        assert run_ids == {"run-a", "run-b"}

    @pytest.mark.asyncio
    async def test_concurrent_backup_operations(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test concurrent backup operations."""

        async def create_backup(i: int) -> None:
            db_path = tmp_path / f"test{i}.db"
            db_path.write_bytes(b"corrupted sqlite data")
            backend = SQLiteBackend(db_path)
            await backend._init_db()

        # Run multiple concurrent backup operations
        tasks = [create_backup(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all backups were created successfully
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 5

    @pytest.mark.asyncio
    async def test_concurrent_database_initialization(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test concurrent database initialization."""
        db_path = tmp_path / "concurrent_test.db"

        async def init_database():
            backend = SQLiteBackend(db_path)
            await backend._ensure_init()
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
    async def test_concurrent_save_operations(self, sqlite_backend_factory) -> None:
        """Test concurrent save operations."""
        backend = sqlite_backend_factory("concurrent_save.db")
        await backend._ensure_init()

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
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
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
    async def test_concurrent_load_operations(self, sqlite_backend_factory) -> None:
        """Test concurrent load operations."""
        backend = sqlite_backend_factory("concurrent_load.db")
        await backend._ensure_init()

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
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
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
    async def test_concurrent_delete_operations(self, sqlite_backend_factory) -> None:
        """Test concurrent delete operations."""
        backend = sqlite_backend_factory("concurrent_delete.db")
        await backend._ensure_init()

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
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
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
    async def test_concurrent_mixed_operations(self, sqlite_backend_factory) -> None:
        """Test concurrent mixed operations (save, load, delete)."""
        backend = sqlite_backend_factory("concurrent_mixed.db")
        await backend._ensure_init()

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
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
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
    async def test_concurrent_backup_with_corruption(
        self, tmp_path: Path, sqlite_backend_factory
    ) -> None:
        """Test concurrent backup operations with corruption scenarios."""

        async def create_corrupted_backup(i: int) -> None:
            db_path = tmp_path / f"corrupt_test{i}.db"
            db_path.write_bytes(b"corrupted sqlite data")
            backend = SQLiteBackend(db_path)
            await backend._init_db()

        # Run multiple concurrent backup operations with corruption
        tasks = [create_corrupted_backup(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all backups were created successfully
        backup_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(backup_files) == 5

        # Verify new databases were created
        for i in range(5):
            db_path = tmp_path / f"corrupt_test{i}.db"
            assert db_path.exists()
            assert db_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, sqlite_backend_factory) -> None:
        """Test concurrent error handling scenarios."""
        backend = sqlite_backend_factory("concurrent_error.db")
        await backend._ensure_init()

        async def error_operation(i: int):
            try:
                # Try to load non-existent state
                result = await backend.load_state(f"non_existent_{i}")
                assert result is None
            except Exception as e:
                # Should not raise exceptions for non-existent states
                assert False, f"Unexpected exception: {e}"

        # Run multiple concurrent error operations
        tasks = [error_operation(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify database is still functional
        workflows = await backend.list_workflows()
        assert isinstance(workflows, list)

    @pytest.mark.asyncio
    async def test_concurrent_performance_under_load(self, sqlite_backend_factory) -> None:
        """Test performance under concurrent load."""
        backend = sqlite_backend_factory("concurrent_perf.db")
        await backend._ensure_init()

        async def performance_operation(i: int):
            # Create state
            state = {
                "run_id": f"perf_test_{i}",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await backend.save_state(f"perf_test_{i}", state)

            # Load state
            loaded = await backend.load_state(f"perf_test_{i}")
            assert loaded is not None

            # Delete state
            await backend.delete_state(f"perf_test_{i}")

        # Run many concurrent operations
        start_time = time.time()
        tasks = [performance_operation(i) for i in range(50)]
        await asyncio.gather(*tasks)
        end_time = time.time()

        # Verify performance is reasonable
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds

        # Verify final state
        workflows = await backend.list_workflows()
        assert len(workflows) == 0  # All should be deleted
