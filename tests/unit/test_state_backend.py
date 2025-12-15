import pytest
from datetime import datetime, timezone
from pydantic import BaseModel

from flujo.state.backends.memory import InMemoryBackend


@pytest.mark.asyncio
async def test_inmemory_backend_roundtrip() -> None:
    backend = InMemoryBackend()
    await backend.save_state("run1", {"foo": 1})
    loaded = await backend.load_state("run1")
    assert loaded == {"foo": 1}
    await backend.delete_state("run1")
    assert await backend.load_state("run1") is None


class Model(BaseModel):
    x: int


@pytest.mark.asyncio
async def test_inmemory_backend_handles_special_types() -> None:
    backend = InMemoryBackend()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    state = {
        "run_id": "run1",
        "pipeline_id": "p",
        "pipeline_name": "p",
        "pipeline_version": "0",
        "current_step_index": 0,
        "pipeline_context": {"dt": now, "val": float("inf"), "model": Model(x=1)},
        "last_step_output": {"nan": float("nan")},
        "status": "running",
        "created_at": now,
        "updated_at": now,
    }

    await backend.save_state("run1", state)
    loaded = await backend.load_state("run1")

    assert loaded is not None
    assert loaded["pipeline_context"]["dt"] == now.isoformat()
    assert loaded["pipeline_context"]["val"] == "inf"
    assert loaded["pipeline_context"]["model"] == {"x": 1}
    assert loaded["last_step_output"]["nan"] == "nan"
