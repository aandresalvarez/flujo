import pytest

from flujo.domain.models import PipelineContext
from flujo.domain.memory import MemoryRecord
from flujo.infra.memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_context_retrieve_with_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryVectorStore()
    await store.add([MemoryRecord(vector=[1.0, 0.0, 0.0], payload="p", metadata={})])

    ctx = PipelineContext()
    setattr(ctx, "memory_store", store)

    results = await ctx.retrieve(query_text=None, query_vector=[1.0, 0.0, 0.0], limit=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_context_retrieve_returns_empty_without_store() -> None:
    ctx = PipelineContext()
    results = await ctx.retrieve(query_text="hello", query_vector=[1.0, 0.0, 0.0])
    assert results == []
