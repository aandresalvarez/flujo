from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from ...domain.memory import (
    MemoryRecord,
    ScoredMemory,
    VectorQuery,
    VectorStoreProtocol,
    _assign_id,
    _cosine_similarity,
)


class NullVectorStore(VectorStoreProtocol):
    """No-op store used as a safe default."""

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        return None

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:
        return []

    async def delete(self, ids: Sequence[str]) -> None:
        return None

    async def close(self) -> None:
        return None


class InMemoryVectorStore(VectorStoreProtocol):
    """In-memory vector store for testing and defaults."""

    def __init__(self) -> None:
        self._store: dict[str, MemoryRecord] = {}
        self._lock = asyncio.Lock()

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        async with self._lock:
            for record in records:
                stored = _assign_id(record)
                record_id = stored.id
                if record_id is None:
                    raise ValueError("MemoryRecord.id must be set after assignment")
                self._store[record_id] = stored

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:
        async with self._lock:
            candidates = list(self._store.values())
        filtered = [
            record
            for record in candidates
            if _metadata_matches(record.metadata, query.filter_metadata)
        ]
        scored: list[ScoredMemory] = []
        for record in filtered:
            score = _cosine_similarity(query.vector, record.vector)
            scored.append(ScoredMemory(record=record, score=score))
        scored.sort(key=lambda item: (-item.score, item.record.id or ""))
        return scored[: max(query.limit, 0)]

    async def delete(self, ids: Sequence[str]) -> None:
        async with self._lock:
            for record_id in ids:
                self._store.pop(record_id, None)

    async def close(self) -> None:
        async with self._lock:
            self._store.clear()


def _metadata_matches(
    metadata: Mapping[str, Any] | None, filter_metadata: Mapping[str, Any] | None
) -> bool:
    if filter_metadata is None or not filter_metadata:
        return True
    if metadata is None:
        return False
    try:
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
    except Exception:
        return False
    return True
