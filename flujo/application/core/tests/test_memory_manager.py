import pytest

from flujo.domain.memory import VectorQuery
from flujo.infra.memory import MemoryManager, NullMemoryManager, InMemoryVectorStore


@pytest.mark.asyncio
async def test_memory_manager_indexes_on_success() -> None:
    store = InMemoryVectorStore()

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    manager = MemoryManager(store=store, embedder=embed, enabled=True, background_task_manager=None)

    class Result:
        success = True
        output = "hello world"

    await manager.index_step_output(step_name="s", result=Result(), context=None)
    records = await store.query(VectorQuery(vector=[0.1, 0.2, 0.3], limit=5))
    assert records


@pytest.mark.asyncio
async def test_memory_manager_skips_when_disabled() -> None:
    store = InMemoryVectorStore()

    async def embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    manager = MemoryManager(
        store=store, embedder=embed, enabled=False, background_task_manager=None
    )

    class Result:
        success = True
        output = "hello world"

    await manager.index_step_output(step_name="s", result=Result(), context=None)
    records = await store.query(VectorQuery(vector=[0.1, 0.2, 0.3], limit=5))
    assert records == []


@pytest.mark.asyncio
async def test_null_memory_manager_noops() -> None:
    manager = NullMemoryManager()

    class Result:
        success = True
        output = "hello world"

    await manager.index_step_output(step_name="s", result=Result(), context=None)
    # No assertion needed; just ensure no exception
