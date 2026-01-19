import pytest

from flujo.domain.memory import MemoryRecord, VectorQuery
from flujo.infra.memory.postgres_vector import PostgresVectorStore


pytestmark = pytest.mark.asyncio


@pytest.mark.skip("Requires pgvector-enabled Postgres test instance")
async def test_postgres_vector_store_add_query_delete(pg_dsn: str) -> None:
    store = PostgresVectorStore(pg_dsn, vector_dimensions=2)
    rec1 = MemoryRecord(vector=[1.0, 0.0], payload={"a": 1}, metadata={"tag": "x"})
    rec2 = MemoryRecord(vector=[0.0, 1.0], payload={"b": 2}, metadata={"tag": "y"})
    await store.add([rec1, rec2])

    results = await store.query(VectorQuery(vector=[1.0, 0.0], limit=5))
    assert results
    assert results[0].record.payload == {"a": 1}

    filtered = await store.query(
        VectorQuery(vector=[0.0, 1.0], limit=5, filter_metadata={"tag": "y"})
    )
    assert len(filtered) == 1
    assert filtered[0].record.payload == {"b": 2}

    await store.delete([rec1.id or ""])
    remaining = await store.query(VectorQuery(vector=[1.0, 0.0], limit=5))
    assert remaining

    await store.close()
