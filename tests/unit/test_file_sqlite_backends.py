from datetime import datetime, timedelta
from pathlib import Path
import asyncio

from pydantic import BaseModel

import aiosqlite

import pytest

from flujo.state.backends.file import FileBackend
from flujo.state.backends.sqlite import SQLiteBackend


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
    now = datetime.utcnow().replace(microsecond=0)
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
    now = datetime.utcnow().replace(microsecond=0)
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
    now = datetime.utcnow().replace(microsecond=0)
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
    now = datetime.utcnow().replace(microsecond=0)
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
        now = datetime.utcnow().replace(microsecond=0)
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

    now = datetime.utcnow().replace(microsecond=0)
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
