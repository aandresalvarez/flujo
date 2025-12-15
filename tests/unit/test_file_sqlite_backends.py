from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio

from pydantic import BaseModel

import aiosqlite

import pytest

from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend

# Mark as very slow to exclude from fast suites
pytestmark = pytest.mark.veryslow


@pytest.fixture(autouse=True)
async def cleanup_sqlite_backends(monkeypatch):
    """Autouse fixture to ensure all SQLiteBackend instances are properly closed.

    This prevents resource leaks that cause 361-second timeouts.
    """
    backends = []
    original_init = SQLiteBackend.__init__

    def tracking_init(self, *args, **kwargs):
        backends.append(self)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "__init__", tracking_init)
    yield
    # Clean up all backends created during the test
    for backend in backends:
        try:
            await backend.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_file_backend_roundtrip(tmp_path: Path) -> None:
    backend = FileBackend(tmp_path)
    state = {"foo": "bar"}
    await backend.save_state("run1", state)
    loaded = await backend.load_state("run1")
    assert loaded == state
    await backend.delete_state("run1")
    assert await backend.load_state("run1") is None


@pytest.mark.asyncio
async def test_file_backend_load_during_delete(tmp_path: Path) -> None:
    backend = FileBackend(tmp_path)
    await backend.save_state("run1", {"foo": 1})

    await asyncio.gather(backend.load_state("run1"), backend.delete_state("run1"))
    # load_state should not raise even if file is deleted concurrently


@pytest.mark.asyncio
async def test_sqlite_backend_roundtrip(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_name": "p",
        "pipeline_version": "0",
        "current_step_index": 1,
        "pipeline_context": {"a": 1},
        "last_step_output": "x",
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }
    await backend.save_state("run1", state)
    loaded = await backend.load_state("run1")
    assert loaded is not None
    assert loaded["pipeline_context"] == {"a": 1}
    assert loaded["last_step_output"] == "x"
    assert loaded["created_at"] == now
    assert loaded["updated_at"] == now
    await backend.delete_state("run1")
    assert await backend.load_state("run1") is None


@pytest.mark.asyncio
async def test_sqlite_backend_migrates_existing_db(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            """
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
            """
        )
        await conn.commit()

    backend = SQLiteBackend(db)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_name": "p",
        "pipeline_version": "0",
        "current_step_index": 0,
        "pipeline_context": {},
        "last_step_output": None,
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }
    await backend.save_state("run1", state)
    loaded = await backend.load_state("run1")
    assert loaded is not None
    assert loaded["pipeline_name"] == "p"


class MyModel(BaseModel):
    x: int


@pytest.mark.asyncio
async def test_backends_serialize_pydantic(tmp_path: Path) -> None:
    fb = FileBackend(tmp_path)
    sb = SQLiteBackend(tmp_path / "s.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_name": "p",
        "pipeline_version": "0",
        "current_step_index": 0,
        "pipeline_context": {"model": MyModel(x=1)},
        "last_step_output": MyModel(x=2),
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }
    await fb.save_state("run1", state)
    await sb.save_state("run1", state)
    loaded_f = await fb.load_state("run1")
    loaded_s = await sb.load_state("run1")
    assert loaded_f["pipeline_context"] == {"model": {"x": 1}}
    assert loaded_s["pipeline_context"] == {"model": {"x": 1}}
    assert loaded_f["last_step_output"] == {"x": 2}
    assert loaded_s["last_step_output"] == {"x": 2}


@pytest.mark.asyncio
async def test_sqlite_backend_admin_queries(tmp_path: Path) -> None:
    """Test list_workflows, get_workflow_stats, get_failed_workflows, cleanup_old_workflows."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    past = now - timedelta(days=1)
    # Insert several states with different statuses and times
    for i, status in enumerate(["running", "completed", "failed", "paused", "failed"]):
        state = {
            "run_id": f"run{i}",
            "pipeline_id": "p",
            "pipeline_name": "p",
            "pipeline_version": "0",
            "current_step_index": i,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": status,
            "created_at": past,
            "updated_at": past,
            "total_steps": 5,
            "error_message": "fail" if status == "failed" else None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await backend.save_state(f"run{i}", state)
    # list_workflows
    all_wf = await backend.list_workflows()
    assert len(all_wf) == 5
    failed = await backend.list_workflows(status="failed")
    assert len(failed) == 2
    # get_workflow_stats
    stats = await backend.get_workflow_stats()
    assert stats["total_workflows"] == 5
    assert stats["status_counts"]["failed"] == 2
    # get_failed_workflows
    failed_wf = await backend.get_failed_workflows(hours_back=48)
    assert len(failed_wf) == 2
    # cleanup_old_workflows
    deleted = await backend.cleanup_old_workflows(days_old=0)
    assert deleted == 5
    # After cleanup, should be empty
    all_wf2 = await backend.list_workflows()
    assert len(all_wf2) == 0


@pytest.mark.asyncio
async def test_sqlite_backend_concurrent(tmp_path: Path) -> None:
    """Test concurrent save/load/delete for SQLiteBackend."""

    async def worker(backend, run_id):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        state = {
            "run_id": run_id,
            "pipeline_id": "p",
            "pipeline_name": "p",
            "pipeline_version": "0",
            "current_step_index": 1,
            "pipeline_context": {"a": 1},
            "last_step_output": "x",
            "status": "running",
            "created_at": now,
            "updated_at": now,
        }
        await backend.save_state(run_id, state)
        loaded = await backend.load_state(run_id)
        assert loaded is not None
        await backend.delete_state(run_id)
        loaded2 = await backend.load_state(run_id)
        assert loaded2 is None

    backend = SQLiteBackend(tmp_path / "state.db")
    await asyncio.gather(*(worker(backend, f"run{i}") for i in range(5)))


@pytest.mark.asyncio
async def test_backends_deserialize_special_types(tmp_path: Path) -> None:
    """Backends should restore special types using safe_deserialize."""
    fb = FileBackend(tmp_path / "fb")
    sb = SQLiteBackend(tmp_path / "s.db")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_name": "p",
        "pipeline_version": "0",
        "current_step_index": 0,
        "pipeline_context": {"dt": now, "val": float("inf")},
        "last_step_output": {"nan": float("nan")},
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }

    await fb.save_state("run1", state)
    await sb.save_state("run1", state)

    loaded_f = await fb.load_state("run1")
    loaded_s = await sb.load_state("run1")

    assert loaded_f is not None and loaded_s is not None
    assert loaded_f["pipeline_context"]["dt"] == now.isoformat()
    assert loaded_s["pipeline_context"]["dt"] == now.isoformat()
    assert loaded_f["pipeline_context"]["val"] == "inf"
    assert loaded_s["pipeline_context"]["val"] == "inf"
    assert loaded_f["last_step_output"]["nan"] == "nan"
    assert loaded_s["last_step_output"]["nan"] == "nan"


@pytest.mark.asyncio
async def test_sqlite_backend_list_workflows_empty_database(tmp_path: Path) -> None:
    """Test list_workflows with an empty database."""
    backend = SQLiteBackend(tmp_path / "state.db")

    # Test list_workflows with no data
    all_workflows = await backend.list_workflows()
    assert all_workflows == []

    # Test list_workflows with status filter on empty database
    running_workflows = await backend.list_workflows(status="running")
    assert running_workflows == []

    # Test list_workflows with pipeline_id filter on empty database
    pipeline_workflows = await backend.list_workflows(pipeline_id="nonexistent")
    assert pipeline_workflows == []

    # Test list_workflows with pagination on empty database
    paginated_workflows = await backend.list_workflows(limit=10, offset=0)
    assert paginated_workflows == []


@pytest.mark.asyncio
async def test_sqlite_backend_get_workflow_stats_empty_database(tmp_path: Path) -> None:
    """Test get_workflow_stats with an empty database."""
    backend = SQLiteBackend(tmp_path / "state.db")

    stats = await backend.get_workflow_stats()

    # Verify expected structure with zero values
    assert stats["total_workflows"] == 0
    assert stats["status_counts"] == {}
    assert stats["recent_workflows_24h"] == 0
    assert stats["average_execution_time_ms"] == 0


@pytest.mark.asyncio
async def test_sqlite_backend_cleanup_old_workflows_no_old_workflows(tmp_path: Path) -> None:
    """Test cleanup_old_workflows when no workflows are old enough to be deleted."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Create recent workflows (less than 1 day old)
    for i in range(3):
        state = {
            "run_id": f"recent_run{i}",
            "pipeline_id": "p",
            "pipeline_name": "p",
            "pipeline_version": "0",
            "current_step_index": i,
            "pipeline_context": {"a": i},
            "last_step_output": f"out{i}",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await backend.save_state(f"recent_run{i}", state)

    # Verify workflows exist
    all_workflows = await backend.list_workflows()
    assert len(all_workflows) == 3

    # Try to cleanup workflows older than 30 days (should delete none)
    deleted_count = await backend.cleanup_old_workflows(days_old=30)
    assert deleted_count == 0

    # Verify workflows still exist
    all_workflows_after = await backend.list_workflows()
    assert len(all_workflows_after) == 3

    # Try to cleanup workflows older than 1 day (should delete none)
    deleted_count_1_day = await backend.cleanup_old_workflows(days_old=1)
    assert deleted_count_1_day == 0

    # Verify workflows still exist
    all_workflows_after_1_day = await backend.list_workflows()
    assert len(all_workflows_after_1_day) == 3


@pytest.mark.asyncio
async def test_sqlite_backend_list_workflows_filter_by_nonexistent_pipeline_id(
    tmp_path: Path,
) -> None:
    """Test filtering by pipeline_id when the ID does not exist."""
    backend = SQLiteBackend(tmp_path / "state.db")
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Create workflows with specific pipeline IDs
    pipeline_ids = ["pipeline_a", "pipeline_b", "pipeline_c"]
    for i, pipeline_id in enumerate(pipeline_ids):
        state = {
            "run_id": f"run_{pipeline_id}_{i}",
            "pipeline_id": pipeline_id,
            "pipeline_name": f"Pipeline {pipeline_id}",
            "pipeline_version": "1.0",
            "current_step_index": i,
            "pipeline_context": {"pipeline": pipeline_id},
            "last_step_output": f"output_{i}",
            "status": "completed",
            "created_at": now,
            "updated_at": now,
            "total_steps": 5,
            "error_message": None,
            "execution_time_ms": 1000 * i,
            "memory_usage_mb": 10.0 * i,
        }
        await backend.save_state(f"run_{pipeline_id}_{i}", state)

    # Verify all workflows exist
    all_workflows = await backend.list_workflows()
    assert len(all_workflows) == 3

    # Test filtering by nonexistent pipeline_id
    nonexistent_workflows = await backend.list_workflows(pipeline_id="nonexistent_pipeline")
    assert nonexistent_workflows == []

    # Test filtering by another nonexistent pipeline_id
    another_nonexistent = await backend.list_workflows(pipeline_id="pipeline_xyz")
    assert another_nonexistent == []

    # Test filtering by empty string pipeline_id (should return all workflows since empty string is falsy)
    empty_pipeline_workflows = await backend.list_workflows(pipeline_id="")
    assert len(empty_pipeline_workflows) == 3  # Empty string is treated as no filter

    # Verify original workflows still exist and can be filtered correctly
    pipeline_a_workflows = await backend.list_workflows(pipeline_id="pipeline_a")
    assert len(pipeline_a_workflows) == 1
    assert pipeline_a_workflows[0]["pipeline_id"] == "pipeline_a"

    pipeline_b_workflows = await backend.list_workflows(pipeline_id="pipeline_b")
    assert len(pipeline_b_workflows) == 1
    assert pipeline_b_workflows[0]["pipeline_id"] == "pipeline_b"


@pytest.mark.asyncio
async def test_sqlite_backend_admin_queries_edge_cases(tmp_path: Path) -> None:
    """Test additional edge cases for admin queries."""
    backend = SQLiteBackend(tmp_path / "state.db")

    # Test get_failed_workflows on empty database
    failed_workflows_empty = await backend.get_failed_workflows(hours_back=24)
    assert failed_workflows_empty == []

    # Test get_failed_workflows with different time ranges on empty database
    for hours in [1, 6, 12, 24, 48, 168]:
        failed_workflows = await backend.get_failed_workflows(hours_back=hours)
        assert failed_workflows == []

    # Test cleanup_old_workflows on empty database
    deleted_empty = await backend.cleanup_old_workflows(days_old=1)
    assert deleted_empty == 0

    # Test cleanup_old_workflows with different time ranges on empty database
    for days in [1, 7, 30, 365]:
        deleted = await backend.cleanup_old_workflows(days_old=days)
        assert deleted == 0
