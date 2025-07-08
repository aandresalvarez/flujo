from datetime import datetime
from pathlib import Path

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
async def test_sqlite_backend_roundtrip(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "state.db")
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_version": "0",
        "current_step_index": 1,
        "pipeline_context": {"a": 1},
        "last_step_output": "x",
        "status": "running",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await backend.save_state("run1", state)
    loaded = await backend.load_state("run1")
    assert loaded is not None
    assert loaded["pipeline_context"] == {"a": 1}
    assert loaded["last_step_output"] == "x"
    await backend.delete_state("run1")
    assert await backend.load_state("run1") is None
