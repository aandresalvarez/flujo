from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TYPE_CHECKING, Sequence

from ...domain.memory import (
    MemoryRecord,
    ScoredMemory,
    VectorQuery,
    VectorStoreProtocol,
    _assign_id,
    _cosine_similarity,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg
    from asyncpg import Pool
else:  # pragma: no cover - runtime checked import
    asyncpg = None
    Pool = Any


class PostgresVectorStore(VectorStoreProtocol):
    """pgvector-backed vector store for production RAG."""

    DEFAULT_VECTOR_DIMENSIONS = 1536

    def __init__(self, dsn: str, *, vector_dimensions: int = DEFAULT_VECTOR_DIMENSIONS) -> None:
        if vector_dimensions <= 0:
            raise ValueError("vector_dimensions must be a positive integer.")
        self._dsn = dsn
        self._vector_dimensions = vector_dimensions
        self._asyncpg: Any | None = None
        self._pool: Pool | None = None
        self._init_lock = asyncio.Lock()
        self._init_done = False

    def _get_asyncpg(self) -> Any:
        if self._asyncpg is None:
            try:
                import importlib

                self._asyncpg = importlib.import_module("asyncpg")
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "asyncpg is required for PostgresVectorStore; install with "
                    "`uv sync --extra postgres` or `pip install .[postgres]`."
                ) from exc
        return self._asyncpg

    async def _ensure_pool(self) -> Pool:
        asyncpg = self._get_asyncpg()
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
                    f"""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        vector vector({self._vector_dimensions}),
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
                existing_dimensions = await self._get_vector_dimensions(conn)
                if existing_dimensions is None:
                    raise RuntimeError(
                        "memories.vector is missing dimensions. "
                        f"Expected vector({self._vector_dimensions}). "
                        "Run migrations or recreate the table with the correct dimension."
                    )
                if existing_dimensions != self._vector_dimensions:
                    raise RuntimeError(
                        "memories.vector dimension mismatch: "
                        f"expected {self._vector_dimensions}, got {existing_dimensions}. "
                        "Update the column type or adjust memory embedding dimensions."
                    )
            self._init_done = True

    async def _get_vector_dimensions(self, conn: Any) -> int | None:
        row = await conn.fetchrow(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS data_type
            FROM pg_attribute a
            WHERE a.attrelid = $1::regclass
              AND a.attname = 'vector'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            "memories",
        )
        if row is None:
            return None
        data_type = row["data_type"]
        if not isinstance(data_type, str):
            return None
        match = re.search(r"vector\((\d+)\)", data_type)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        if data_type.strip() == "vector":
            return None
        return None

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        await self._init()
        if not records:
            return
        assigned = [_assign_id(r) for r in records]

        # Validate vector dimensions (pgvector table uses vector(N))
        for rec in assigned:
            if rec.vector is not None and len(rec.vector) != self._vector_dimensions:
                raise ValueError(
                    "Vector dimension mismatch: expected "
                    f"{self._vector_dimensions}, got {len(rec.vector)}. "
                    f"Record ID: {rec.id}."
                )

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
        if len(query.vector) != self._vector_dimensions:
            raise ValueError(
                "Vector dimension mismatch: expected "
                f"{self._vector_dimensions}, got {len(query.vector)}."
            )
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
