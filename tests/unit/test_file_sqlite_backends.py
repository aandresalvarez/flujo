from datetime import datetime
from pathlib import Path
import asyncio

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
