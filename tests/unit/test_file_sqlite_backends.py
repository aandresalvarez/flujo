from datetime import datetime
from pathlib import Path
import asyncio
from typing import Any

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


@pytest.mark.skip(reason="Custom serializer_default is no longer supported in FileBackend.")
@pytest.mark.asyncio
async def test_serializer_default_override(tmp_path: Path) -> None:
    def handler(obj: Any) -> Any:
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        raise TypeError

    backend = FileBackend(tmp_path, serializer_default=handler)
    await backend.save_state("r", {"foo": 1 + 2j})
    loaded = await backend.load_state("r")
    assert loaded == {"foo": {"real": 1.0, "imag": 2.0}}
