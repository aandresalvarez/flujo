"""Integration tests for PostgresBackend.

These tests require asyncpg and optionally testcontainers for Docker-based Postgres.
Marked as integration and slow tests per PRD requirements.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
import pytest

from flujo.state.backends.postgres import PostgresBackend
from flujo.type_definitions.common import JSONObject


# Check if asyncpg is available
_asyncpg_available = importlib.util.find_spec("asyncpg") is not None

# Check if testcontainers is available
_testcontainers_available = importlib.util.find_spec("testcontainers") is not None

# Mark all tests as integration and slow per PRD
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _asyncpg_available, reason="asyncpg not installed"),
]


@pytest.fixture
async def postgres_backend() -> PostgresBackend:
    """Create a PostgresBackend instance for testing.

    Uses testcontainers if available, otherwise requires FLUJO_TEST_POSTGRES_URI env var.
    """
    if _testcontainers_available:
        try:
            from testcontainers.postgres import PostgresContainer

            with PostgresContainer("postgres:15") as postgres:
                dsn = postgres.get_connection_url().replace("postgresql://", "postgres://")
                backend = PostgresBackend(dsn, auto_migrate=True)
                yield backend
                await backend.shutdown()
        except Exception as e:
            pytest.skip(f"Failed to start testcontainers Postgres: {e}")
    else:
        import os

        test_uri = os.environ.get("FLUJO_TEST_POSTGRES_URI")
        if not test_uri:
            pytest.skip(
                "testcontainers not available and FLUJO_TEST_POSTGRES_URI not set. "
                "Install testcontainers or set FLUJO_TEST_POSTGRES_URI environment variable."
            )
        backend = PostgresBackend(test_uri, auto_migrate=True)
        yield backend
        await backend.shutdown()


@pytest.mark.asyncio
async def test_postgres_backend_save_and_load(postgres_backend: PostgresBackend) -> None:
    """Test basic save and load operations."""
    run_id = "test_run_1"
    state: JSONObject = {
        "run_id": run_id,
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {"key": "value", "nested": {"data": 123}},
        "last_step_output": {"result": "success"},
        "step_history": [],
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "total_steps": 1,
    }

    await postgres_backend.save_state(run_id, state)
    loaded = await postgres_backend.load_state(run_id)

    assert loaded is not None
    assert loaded["run_id"] == run_id
    assert loaded["pipeline_id"] == "test_pipeline"
    assert loaded["pipeline_context"]["key"] == "value"
    assert loaded["pipeline_context"]["nested"]["data"] == 123


@pytest.mark.asyncio
async def test_postgres_backend_delete(postgres_backend: PostgresBackend) -> None:
    """Test delete operation."""
    run_id = "test_run_delete"
    state: JSONObject = {
        "run_id": run_id,
        "pipeline_id": "test_pipeline",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "step_history": [],
        "status": "running",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "total_steps": 1,
    }

    await postgres_backend.save_state(run_id, state)
    loaded = await postgres_backend.load_state(run_id)
    assert loaded is not None

    await postgres_backend.delete_state(run_id)
    loaded_after = await postgres_backend.load_state(run_id)
    assert loaded_after is None


@pytest.mark.asyncio
async def test_postgres_backend_list_workflows(postgres_backend: PostgresBackend) -> None:
    """Test list_workflows method."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=1)

    # Insert multiple workflows with different statuses
    for i, status in enumerate(["running", "completed", "failed", "paused", "failed"]):
        state: JSONObject = {
            "run_id": f"run_{i}",
            "pipeline_id": "pipeline_1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": i,
            "pipeline_context": {"index": i},
            "step_history": [],
            "status": status,
            "created_at": past,
            "updated_at": past,
            "total_steps": 5,
            "error_message": "fail" if status == "failed" else None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await postgres_backend.save_state(f"run_{i}", state)

    # Test listing all workflows
    all_workflows = await postgres_backend.list_workflows()
    assert len(all_workflows) == 5

    # Test filtering by status
    failed_workflows = await postgres_backend.list_workflows(status="failed")
    assert len(failed_workflows) == 2

    # Test filtering by pipeline_id
    pipeline_workflows = await postgres_backend.list_workflows(pipeline_id="pipeline_1")
    assert len(pipeline_workflows) == 5

    # Test limit
    limited = await postgres_backend.list_workflows(limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_postgres_backend_list_runs(postgres_backend: PostgresBackend) -> None:
    """Test list_runs method for lens CLI compatibility."""
    now = datetime.now(timezone.utc)

    # Create run records
    for i in range(3):
        run_data: JSONObject = {
            "run_id": f"run_{i}",
            "pipeline_id": "pipeline_1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "status": "completed" if i % 2 == 0 else "running",
            "created_at": now,
            "updated_at": now,
        }
        await postgres_backend.save_run_start(run_data)

    runs = await postgres_backend.list_runs()
    assert len(runs) == 3

    # Test filtering
    completed_runs = await postgres_backend.list_runs(status="completed")
    assert len(completed_runs) == 2

    # Test limit
    limited = await postgres_backend.list_runs(limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_postgres_backend_get_workflow_stats(postgres_backend: PostgresBackend) -> None:
    """Test get_workflow_stats method."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=1)

    # Insert workflows with different statuses
    for i, status in enumerate(["running", "completed", "failed", "paused", "failed"]):
        state: JSONObject = {
            "run_id": f"run_{i}",
            "pipeline_id": "pipeline_1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": i,
            "pipeline_context": {},
            "step_history": [],
            "status": status,
            "created_at": past if i < 2 else now,  # Some old, some recent
            "updated_at": past if i < 2 else now,
            "total_steps": 5,
            "execution_time_ms": 1000 * i,
        }
        await postgres_backend.save_state(f"run_{i}", state)

    stats = await postgres_backend.get_workflow_stats()

    assert stats["total_workflows"] == 5
    assert stats["status_counts"]["failed"] == 2
    assert stats["status_counts"]["running"] == 1
    assert stats["status_counts"]["completed"] == 1
    assert stats["status_counts"]["paused"] == 1
    assert stats["recent_workflows_24h"] >= 3  # At least the recent ones
    assert stats["average_execution_time_ms"] > 0


@pytest.mark.asyncio
async def test_postgres_backend_get_failed_workflows(postgres_backend: PostgresBackend) -> None:
    """Test get_failed_workflows method."""
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=12)  # Recent failure
    old = now - timedelta(days=2)  # Old failure

    # Create recent failed workflow
    state1: JSONObject = {
        "run_id": "run_recent_failed",
        "pipeline_id": "pipeline_1",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "step_history": [],
        "status": "failed",
        "created_at": recent,
        "updated_at": recent,
        "total_steps": 1,
        "error_message": "Recent error",
    }
    await postgres_backend.save_state("run_recent_failed", state1)

    # Create old failed workflow (should not be included)
    state2: JSONObject = {
        "run_id": "run_old_failed",
        "pipeline_id": "pipeline_1",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "step_history": [],
        "status": "failed",
        "created_at": old,
        "updated_at": old,
        "total_steps": 1,
        "error_message": "Old error",
    }
    await postgres_backend.save_state("run_old_failed", state2)

    # Get failed workflows from last 24 hours
    failed = await postgres_backend.get_failed_workflows(hours_back=24)
    assert len(failed) == 1
    assert failed[0]["run_id"] == "run_recent_failed"
    assert failed[0]["error_message"] == "Recent error"


@pytest.mark.asyncio
async def test_postgres_backend_cleanup_old_workflows(postgres_backend: PostgresBackend) -> None:
    """Test cleanup_old_workflows method."""
    now = datetime.now(timezone.utc)
    recent = now - timedelta(days=1)
    old = now - timedelta(days=35)

    # Create recent workflow
    state1: JSONObject = {
        "run_id": "run_recent",
        "pipeline_id": "pipeline_1",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "step_history": [],
        "status": "completed",
        "created_at": recent,
        "updated_at": recent,
        "total_steps": 1,
    }
    await postgres_backend.save_state("run_recent", state1)

    # Create old workflow
    state2: JSONObject = {
        "run_id": "run_old",
        "pipeline_id": "pipeline_1",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "current_step_index": 0,
        "pipeline_context": {},
        "step_history": [],
        "status": "completed",
        "created_at": old,
        "updated_at": old,
        "total_steps": 1,
    }
    await postgres_backend.save_state("run_old", state2)

    # Cleanup workflows older than 30 days
    deleted = await postgres_backend.cleanup_old_workflows(days_old=30)
    assert deleted == 1

    # Verify old workflow is gone
    assert await postgres_backend.load_state("run_old") is None
    # Verify recent workflow still exists
    assert await postgres_backend.load_state("run_recent") is not None


@pytest.mark.asyncio
async def test_postgres_backend_trace_operations(postgres_backend: PostgresBackend) -> None:
    """Test trace save and load operations."""
    run_id = "test_trace_run"
    trace: JSONObject = {
        "span_id": "root",
        "name": "pipeline",
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "status": "completed",
        "attributes": {"test": "data"},
    }

    await postgres_backend.save_trace(run_id, trace)
    loaded_trace = await postgres_backend.get_trace(run_id)

    assert loaded_trace is not None
    assert loaded_trace["span_id"] == "root"
    assert loaded_trace["name"] == "pipeline"
    assert loaded_trace["attributes"]["test"] == "data"

    spans = await postgres_backend.get_spans(run_id)
    assert len(spans) == 1
    assert spans[0]["span_id"] == "root"
    assert spans[0]["name"] == "pipeline"

    stats = await postgres_backend.get_span_statistics()
    assert stats["total_spans"] >= 1


@pytest.mark.asyncio
async def test_postgres_backend_run_details(postgres_backend: PostgresBackend) -> None:
    """Test get_run_details and list_run_steps methods."""
    run_id = "test_run_details"
    now = datetime.now(timezone.utc)

    # Save run start
    run_data: JSONObject = {
        "run_id": run_id,
        "pipeline_id": "pipeline_1",
        "pipeline_name": "Test Pipeline",
        "pipeline_version": "1.0",
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }
    await postgres_backend.save_run_start(run_data)

    # Save step results
    for i in range(3):
        step_data: JSONObject = {
            "run_id": run_id,
            "step_name": f"step_{i}",
            "step_index": i,
            "status": "completed",
            "output": {"result": f"output_{i}"},
            "created_at": now,
        }
        await postgres_backend.save_step_result(step_data)

    # Get run details
    details = await postgres_backend.get_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert details["pipeline_name"] == "Test Pipeline"

    # Get run steps
    steps = await postgres_backend.list_run_steps(run_id)
    assert len(steps) == 3
    assert steps[0]["step_name"] == "step_0"
    assert steps[1]["step_name"] == "step_1"
    assert steps[2]["step_name"] == "step_2"

    # Save run end
    end_data: JSONObject = {
        "status": "completed",
        "updated_at": now,
        "execution_time_ms": 5000,
        "total_steps": 3,
    }
    await postgres_backend.save_run_end(run_id, end_data)

    # Verify updated details
    updated_details = await postgres_backend.get_run_details(run_id)
    assert updated_details is not None
    assert updated_details["status"] == "completed"
    assert updated_details["execution_time_ms"] == 5000
    assert updated_details["total_steps"] == 3


@pytest.mark.asyncio
async def test_postgres_backend_auto_migrate_disabled() -> None:
    """Test that backend raises error when auto_migrate is False and schema doesn't exist."""
    if not _testcontainers_available:
        pytest.skip("Requires testcontainers for isolated database")
    from testcontainers.postgres import PostgresContainer

    # Use a fresh container to guarantee an empty schema
    with PostgresContainer("postgres:15") as postgres:
        dsn = postgres.get_connection_url().replace("postgresql://", "postgres://")
        backend = PostgresBackend(dsn, auto_migrate=False)
        try:
            with pytest.raises(RuntimeError):
                await backend.load_state("missing")
        finally:
            await backend.shutdown()


@pytest.mark.asyncio
async def test_postgres_backend_connection_pooling(postgres_backend: PostgresBackend) -> None:
    """Test that connection pooling works correctly."""

    # Create multiple concurrent operations
    async def save_state(i: int) -> None:
        state: JSONObject = {
            "run_id": f"concurrent_run_{i}",
            "pipeline_id": "pipeline_1",
            "pipeline_name": "Test Pipeline",
            "pipeline_version": "1.0",
            "current_step_index": 0,
            "pipeline_context": {"index": i},
            "step_history": [],
            "status": "running",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "total_steps": 1,
        }
        await postgres_backend.save_state(f"concurrent_run_{i}", state)

    # Run 10 concurrent saves
    await asyncio.gather(*[save_state(i) for i in range(10)])

    # Verify all states were saved
    all_workflows = await postgres_backend.list_workflows()
    concurrent_workflows = [
        wf for wf in all_workflows if wf["run_id"].startswith("concurrent_run_")
    ]
    assert len(concurrent_workflows) == 10
