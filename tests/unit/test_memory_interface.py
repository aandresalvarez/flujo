from typing import Sequence

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.runtime_builder import FlujoRuntimeBuilder
from flujo.domain.memory import MemoryRecord, VectorQuery
from flujo.infra.memory import InMemoryVectorStore, NullVectorStore


class DummyVectorStore(NullVectorStore):
    """Test double that records usage without altering behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.called = False

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        self.called = True
        return await super().add(records)


@pytest.mark.asyncio
async def test_in_memory_store_add_query_delete() -> None:
    store = InMemoryVectorStore()
    r1 = MemoryRecord(vector=[1.0, 0.0], payload="a")
    r2 = MemoryRecord(vector=[0.0, 1.0], payload="b")

    await store.add([r1, r2])

    result = await store.query(VectorQuery(vector=[1.0, 0.0], limit=2))
    assert [item.record.payload for item in result] == ["a", "b"]
    first_id = result[0].record.id
    assert first_id is not None

    await store.delete([first_id])
    result_after = await store.query(VectorQuery(vector=[1.0, 0.0], limit=2))
    assert [item.record.payload for item in result_after] == ["b"]


@pytest.mark.asyncio
async def test_in_memory_store_respects_metadata_filter() -> None:
    store = InMemoryVectorStore()
    r1 = MemoryRecord(vector=[0.0, 1.0], payload="alpha", metadata={"topic": "x"})
    r2 = MemoryRecord(vector=[0.0, 1.0], payload="beta", metadata={"topic": "y"})

    await store.add([r1, r2])

    hits = await store.query(
        VectorQuery(vector=[0.0, 1.0], limit=5, filter_metadata={"topic": "x"})
    )
    assert len(hits) == 1
    assert hits[0].record.payload == "alpha"


@pytest.mark.asyncio
async def test_null_store_is_noop() -> None:
    store = NullVectorStore()
    await store.add([MemoryRecord(vector=[1.0], payload="noop")])
    result = await store.query(VectorQuery(vector=[1.0], limit=1))
    assert result == []
    await store.delete(["unused"])
    await store.close()


def test_runtime_builder_defaults_to_null_store() -> None:
    deps = FlujoRuntimeBuilder().build()
    assert isinstance(deps.memory_store, NullVectorStore)


def test_runtime_builder_accepts_custom_store() -> None:
    custom_store = DummyVectorStore()
    deps = FlujoRuntimeBuilder().build(memory_store=custom_store)
    assert deps.memory_store is custom_store
    assert isinstance(deps.memory_store, DummyVectorStore)


def test_executor_core_exposes_memory_store() -> None:
    custom_store = DummyVectorStore()
    deps = FlujoRuntimeBuilder().build(memory_store=custom_store)
    core = ExecutorCore(deps=deps)
    assert core.memory_store is custom_store
