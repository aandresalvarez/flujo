import asyncio

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


@pytest.mark.asyncio
async def test_memory_manager_tasks_awaited_on_shutdown() -> None:
    """Shutdown should wait for in-flight indexing tasks to finish."""

    store = InMemoryVectorStore()
    started = asyncio.Event()
    released = asyncio.Event()
    embed_calls: list[str] = []

    async def embed(texts: list[str]) -> list[list[float]]:
        embed_calls.extend(texts)
        started.set()
        await released.wait()
        return [[0.1, 0.2, 0.3] for _ in texts]

    manager = MemoryManager(store=store, embedder=embed, enabled=True, background_task_manager=None)

    class Result:
        success = True
        output = "hello world"

    # Kick off indexing without awaiting completion yet
    indexing_task = asyncio.create_task(
        manager.index_step_output(step_name="s", result=Result(), context=None)
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)
    shutdown_task = asyncio.create_task(manager.close())
    await asyncio.sleep(0.05)  # give close() time to gather pending tasks

    # Allow embedder to finish and ensure shutdown waits
    released.set()
    await asyncio.wait_for(indexing_task, timeout=1.0)
    await asyncio.wait_for(shutdown_task, timeout=1.0)

    assert embed_calls == ["hello world"]
    assert manager._pending_tasks == []
