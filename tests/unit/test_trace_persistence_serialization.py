from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

import pytest


class _Level(Enum):
    LOW = "low"


@dataclass
class _DC:
    x: int


@pytest.mark.asyncio
async def test_trace_attributes_use_persistence_serializer(sqlite_backend_factory) -> None:
    backend = sqlite_backend_factory("trace_serialization.db")
    run_id = "trace_serialization_run"

    now = datetime.now(timezone.utc)
    await backend.save_run_start(
        {
            "run_id": run_id,
            "pipeline_id": str(uuid4()),
            "pipeline_name": "p",
            "pipeline_version": "1",
            "status": "running",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
    )

    trace_data = {
        "span_id": "root",
        "name": "pipeline",
        "status": "completed",
        "start_time": 1.0,
        "end_time": 2.0,
        "attributes": {
            "when": now,
            "uuid": uuid4(),
            "level": _Level.LOW,
            "dc": _DC(3),
            "data": b"hello",
            "inf": float("inf"),
            "nan": float("nan"),
            "set": {1, 2},
        },
        "children": [],
    }
    await backend.save_trace(run_id, trace_data)  # type: ignore[arg-type]

    loaded = await backend.get_trace(run_id)
    assert loaded is not None
    attrs = loaded["attributes"]
    assert attrs["when"] == now.isoformat()
    assert isinstance(attrs["uuid"], str) and attrs["uuid"]
    assert attrs["level"] == "low"
    assert attrs["dc"] == {"x": 3}
    assert attrs["data"] == "hello"
    assert attrs["inf"] == "inf"
    assert attrs["nan"] == "nan"
    assert sorted(attrs["set"]) == [1, 2]
