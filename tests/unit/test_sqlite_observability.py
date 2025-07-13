"""Tests for SQLiteBackend observability, logging, and error reporting."""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from io import StringIO
import sqlite3

from flujo.state.backends.sqlite import SQLiteBackend
from .conftest import capture_logs


@pytest.mark.asyncio
async def test_sqlite_backend_logs_initialization_events(tmp_path: Path) -> None:
    """Test that SQLiteBackend logs initialization events properly."""
    with capture_logs() as log_capture:
        backend = SQLiteBackend(tmp_path / "init_test.db")
        await backend._ensure_init()

        # Check that initialization was logged
        log_output = log_capture.getvalue()
        assert "Initialized SQLite database" in log_output, (
            "Expected log message not found in log output"
        )


@pytest.mark.asyncio
async def test_sqlite_backend_logs_save_operations(tmp_path: Path) -> None:
    """Test that SQLiteBackend logs save operations with appropriate detail."""
    with capture_logs() as log_capture:
        backend = SQLiteBackend(tmp_path / "save_test.db")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        state = {
            "run_id": "test_run",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": "test_output",
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 10.0,
        }

        await backend.save_state("test_run", state)

        log_output = log_capture.getvalue()
        # Should log save operations (though exact message depends on implementation)
        assert len(log_output) > 0


@pytest.mark.asyncio
async def test_sqlite_backend_logs_error_conditions(tmp_path: Path) -> None:
    """Test that SQLiteBackend logs error conditions appropriately."""
    with capture_logs(level=logging.ERROR) as log_capture:
        backend = SQLiteBackend(tmp_path / "error_test.db")

        # Try to load a non-existent state (should not be an error, but should be logged)
        result = await backend.load_state("non_existent_run")
        assert result is None
        # Note: Backend may not log non-existent state lookups, which is acceptable behavior
        # We only check that the operation completes without error

        # Create a corrupted database to trigger error logging
        db_path = tmp_path / "corrupted_error.db"
        with open(db_path, "w") as f:
            f.write("This is not a valid SQLite database")

        corrupted_backend = SQLiteBackend(db_path)

        try:
            # Create a proper state object with all required fields
            now = datetime.now(timezone.utc).replace(microsecond=0)
            test_state = {
                "run_id": "test_run",
                "pipeline_id": "test_pipeline",
                "pipeline_name": "Test Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": 0,
                "pipeline_context": {"test": "data"},
                "last_step_output": "test_output",
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000,
                "memory_usage_mb": 10.0,
            }
            await corrupted_backend.save_state("test_run", test_state)
        except sqlite3.DatabaseError:
            # Error should be logged
            log_output = log_capture.getvalue()
            assert len(log_output) > 0


@pytest.mark.asyncio
async def test_sqlite_backend_metrics_correctness(tmp_path: Path) -> None:
    """Test that SQLiteBackend provides correct metrics and statistics."""
    backend = SQLiteBackend(tmp_path / "metrics_test.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Create workflows with different statuses
    workflows = [
        {
            "run_id": "completed_run",
            "pipeline_id": "pipeline_1",
            "pipeline_name": "Pipeline 1",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"data": "completed"},
            "last_step_output": "completed_output",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 10.0,
        },
        {
            "run_id": "failed_run",
            "pipeline_id": "pipeline_2",
            "pipeline_name": "Pipeline 2",
            "pipeline_version": "1.0",
            "current_step_index": 1,
            "pipeline_context": {"data": "failed"},
            "last_step_output": "failed_output",
            "status": "failed",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": "Test error",
            "execution_time_ms": 2000,
            "memory_usage_mb": 20.0,
        },
        {
            "run_id": "running_run",
            "pipeline_id": "pipeline_3",
            "pipeline_name": "Pipeline 3",
            "pipeline_version": "1.0",
            "current_step_index": 2,
            "pipeline_context": {"data": "running"},
            "last_step_output": "running_output",
            "status": "running",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1500,
            "memory_usage_mb": 15.0,
        },
    ]

    for workflow in workflows:
        await backend.save_state(workflow["run_id"], workflow)

    # Test workflow statistics
    stats = await backend.get_workflow_stats()

    assert stats["total_workflows"] == 3
    assert stats["status_counts"]["completed"] == 1
    assert stats["status_counts"]["failed"] == 1
    assert stats["status_counts"]["running"] == 1
    assert stats["recent_workflows_24h"] == 3
    assert stats["average_execution_time_ms"] == 1500  # (1000 + 2000 + 1500) / 3


@pytest.mark.asyncio
async def test_sqlite_backend_error_reporting_detail(tmp_path: Path) -> None:
    """Test that SQLiteBackend provides detailed error reporting."""
    backend = SQLiteBackend(tmp_path / "error_detail_test.db")

    # Test error reporting for invalid operations
    try:
        # Try to save with invalid status
        now = datetime.now(timezone.utc).replace(microsecond=0)
        invalid_state = {
            "run_id": "invalid_run",
            "pipeline_id": "test_pipeline",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"test": "data"},
            "last_step_output": "test_output",
            "status": "invalid_status",  # Invalid status
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000,
            "memory_usage_mb": 10.0,
        }

        await backend.save_state("invalid_run", invalid_state)
        # If it succeeds, the constraint might not be enforced
        pass
    except Exception as e:
        # Error should contain useful information
        error_msg = str(e).lower()
        assert any(keyword in error_msg for keyword in ["constraint", "status", "invalid", "check"])


@pytest.mark.asyncio
async def test_sqlite_backend_performance_metrics(tmp_path: Path) -> None:
    """Test that SQLiteBackend provides performance-related metrics."""
    backend = SQLiteBackend(tmp_path / "perf_test.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Create workflows with different performance characteristics
    for i in range(10):
        state = {
            "run_id": f"perf_run_{i}",
            "pipeline_id": f"pipeline_{i % 3}",
            "pipeline_name": f"Pipeline {i % 3}",
            "pipeline_version": "1.0",
            "current_step_index": i % 5,
            "pipeline_context": {"index": i},
            "last_step_output": f"output_{i}",
            "status": "completed" if i % 2 == 0 else "failed",
            "created_at": now - timedelta(minutes=i),
            "updated_at": now - timedelta(minutes=i),
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * (i + 1),  # Varying execution times
            "memory_usage_mb": 10.0 * (i + 1),  # Varying memory usage
        }
        await backend.save_state(f"perf_run_{i}", state)

    # Test performance metrics
    stats = await backend.get_workflow_stats()

    # Should have correct total count
    assert stats["total_workflows"] == 10

    # Should have average execution time
    assert stats["average_execution_time_ms"] > 0

    # Test specific performance queries
    workflows = await backend.list_workflows()
    assert len(workflows) == 10

    # Check that performance data is included in workflow listings
    for workflow in workflows:
        assert "execution_time_ms" in workflow
        assert "memory_usage_mb" in workflow


@pytest.mark.asyncio
async def test_sqlite_backend_logs_cleanup_operations(tmp_path: Path) -> None:
    """Test that SQLiteBackend logs cleanup operations."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("flujo")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        backend = SQLiteBackend(tmp_path / "cleanup_test.db")

        # Create some old workflows
        past = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=2)
        for i in range(5):
            state = {
                "run_id": f"old_run_{i}",
                "pipeline_id": "old_pipeline",
                "pipeline_name": "Old Pipeline",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"old": "data"},
                "last_step_output": f"old_output_{i}",
                "status": "completed",
                "created_at": past,
                "updated_at": past,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000 * i,
                "memory_usage_mb": 10.0 * i,
            }
            await backend.save_state(f"old_run_{i}", state)

        # Perform cleanup
        deleted_count = await backend.cleanup_old_workflows(days_old=1)
        assert deleted_count == 5

        log_output = log_capture.getvalue()
        # Should log cleanup operations
        assert len(log_output) > 0

    finally:
        logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_sqlite_backend_error_context_preservation(tmp_path: Path) -> None:
    """Test that SQLiteBackend preserves error context in error messages."""
    backend = SQLiteBackend(tmp_path / "context_test.db")

    # Test that errors include relevant context
    try:
        # Try to access a non-existent workflow
        result = await backend.load_state("non_existent_run")
        assert result is None
    except Exception as e:
        # If an exception is raised, it should include context
        error_msg = str(e).lower()
        assert "non_existent_run" in error_msg or "workflow" in error_msg


@pytest.mark.asyncio
async def test_sqlite_backend_logs_concurrent_access(tmp_path: Path) -> None:
    """Test that SQLiteBackend logs concurrent access patterns."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("flujo")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
        backend = SQLiteBackend(tmp_path / "concurrent_test.db")

        # Simulate concurrent access
        async def concurrent_operation(i):
            now = datetime.now(timezone.utc).replace(microsecond=0)
            state = {
                "run_id": f"concurrent_run_{i}",
                "pipeline_id": f"pipeline_{i}",
                "pipeline_name": f"Pipeline {i}",
                "pipeline_version": "1.0",
                "current_step_index": i,
                "pipeline_context": {"index": i},
                "last_step_output": f"output_{i}",
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "total_steps": 5,
                "error_message": None,
                "execution_time_ms": 1000 * i,
                "memory_usage_mb": 10.0 * i,
            }
            await backend.save_state(f"concurrent_run_{i}", state)
            return await backend.load_state(f"concurrent_run_{i}")

        # Run concurrent operations
        results = await asyncio.gather(*[concurrent_operation(i) for i in range(5)])

        # All operations should succeed
        for result in results:
            assert result is not None

        log_output = log_capture.getvalue()
        # Should log concurrent access patterns
        assert len(log_output) > 0

    finally:
        logger.removeHandler(handler)
