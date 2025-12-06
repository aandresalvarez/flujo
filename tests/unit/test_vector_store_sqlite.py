from pathlib import Path

import pytest

from flujo.domain.memory import MemoryRecord, VectorQuery
from flujo.infra.memory.sqlite_vector import SQLiteVectorStore


@pytest.mark.asyncio
async def test_sqlite_vector_store_add_query_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    store = SQLiteVectorStore(str(db_path))

    rec1 = MemoryRecord(vector=[1.0, 0.0], payload={"a": 1}, metadata={"tag": "x"})
    rec2 = MemoryRecord(vector=[0.0, 1.0], payload={"b": 2}, metadata={"tag": "y"})
    await store.add([rec1, rec2])

    results = await store.query(VectorQuery(vector=[1.0, 0.0], limit=5))
    assert len(results) == 2
    assert results[0].record.payload == {"a": 1}

    filtered = await store.query(
        VectorQuery(vector=[0.0, 1.0], limit=5, filter_metadata={"tag": "y"})
    )
    assert len(filtered) == 1
    assert filtered[0].record.payload == {"b": 2}

    await store.delete([rec1.id or ""])
    remaining = await store.query(VectorQuery(vector=[1.0, 0.0], limit=5))
    assert len(remaining) == 1
    assert remaining[0].record.id == rec2.id


@pytest.mark.asyncio
async def test_sqlite_vector_store_handles_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    store = SQLiteVectorStore(str(db_path))
    results = await store.query(VectorQuery(vector=[0.0], limit=3))
    assert results == []
