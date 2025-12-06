from __future__ import annotations

import asyncio
import json
from typing import Sequence

import asyncpg

from ...domain.memory import (
    MemoryRecord,
    ScoredMemory,
    VectorQuery,
    VectorStoreProtocol,
    _assign_id,
    _cosine_similarity,
)


class PostgresVectorStore(VectorStoreProtocol):
    """pgvector-backed vector store for production RAG."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._init_lock = asyncio.Lock()
        self._init_done = False

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=5)
        return self._pool

    async def _init(self) -> None:
        if self._init_done:
            return
        async with self._init_lock:
            if self._init_done:
                return
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        vector vector,
                        payload JSONB,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding
                    ON memories
                    USING hnsw (vector vector_cosine_ops);
                    """
                )
            self._init_done = True

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        await self._init()
        if not records:
            return
        assigned = [_assign_id(r) for r in records]
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memories (id, vector, payload, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    vector = EXCLUDED.vector,
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at;
                """,
                [
                    (
                        rec.id,
                        rec.vector,
                        json.dumps(rec.payload) if rec.payload is not None else None,
                        json.dumps(rec.metadata) if rec.metadata is not None else None,
                        rec.timestamp,
                    )
                    for rec in assigned
                ],
            )

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:
        await self._init()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, vector, payload, metadata, created_at
                FROM memories
                ORDER BY vector <=> $1
                LIMIT $2;
                """,
                query.vector,
                max(query.limit, 0),
            )
        results: list[ScoredMemory] = []
        for row in rows:
            metadata = row["metadata"]
            if query.filter_metadata:
                if not metadata:
                    continue
                matches = True
                for k, v in query.filter_metadata.items():
                    if metadata.get(k) != v:
                        matches = False
                        break
                if not matches:
                    continue
            record = MemoryRecord(
                id=row["id"],
                vector=row["vector"],
                payload=row["payload"],
                metadata=metadata,
                timestamp=row["created_at"],
            )
            score = _cosine_similarity(query.vector, record.vector)
            results.append(ScoredMemory(record=record, score=score))
        results.sort(key=lambda item: (-item.score, item.record.id or ""))
        return results

    async def delete(self, ids: Sequence[str]) -> None:
        await self._init()
        if not ids:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany("DELETE FROM memories WHERE id = $1", [(i,) for i in ids])

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._init_done = False
