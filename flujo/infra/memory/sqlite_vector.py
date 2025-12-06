from __future__ import annotations

import asyncio
import heapq
import json
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ...domain.memory import (
    MemoryRecord,
    ScoredMemory,
    VectorQuery,
    VectorStoreProtocol,
    _assign_id,
    _cosine_similarity,
)

FETCH_CHUNK_SIZE = 512


def _ensure_directory(db_path: Path) -> None:
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _serialize_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    count = len(blob) // struct.calcsize("f")
    return list(struct.unpack(f"{count}f", blob))


class SQLiteVectorStore(VectorStoreProtocol):
    """Lightweight SQLite-backed vector store (no C extensions)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        _ensure_directory(self._db_path)
        self._lock = asyncio.Lock()
        self._init_done = False

    async def _init(self) -> None:
        if self._init_done:
            return
        async with self._lock:
            if self._init_done:
                return
            await asyncio.to_thread(self._init_sync)
            self._init_done = True

    def _init_sync(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    payload TEXT,
                    metadata TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    async def add(self, records: Sequence[MemoryRecord]) -> None:
        await self._init()
        if not records:
            return
        assigned = [_assign_id(r) for r in records]
        payloads = [
            (
                rec.id,
                _serialize_vector(rec.vector),
                json.dumps(rec.payload, default=str) if rec.payload is not None else None,
                json.dumps(rec.metadata, default=str) if rec.metadata is not None else None,
                rec.timestamp.isoformat(),
            )
            for rec in assigned
        ]
        await asyncio.to_thread(self._add_sync, payloads)

    def _add_sync(
        self, payloads: list[tuple[str | None, bytes, str | None, str | None, str]]
    ) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO memories (id, vector, payload, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                payloads,
            )
            conn.commit()
        finally:
            conn.close()

    async def query(self, query: VectorQuery) -> list[ScoredMemory]:
        await self._init()
        limit = max(query.limit, 0)
        if limit == 0:
            return []

        return await asyncio.to_thread(self._query_top_k_sync, query, limit)

    def _query_top_k_sync(self, query: VectorQuery, limit: int) -> list[ScoredMemory]:
        """Compute top-k cosine similarities without loading entire table into memory."""
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute("SELECT id, vector, payload, metadata, created_at FROM memories")
            heap: list[tuple[float, ScoredMemory]] = []
            while True:
                rows = cur.fetchmany(FETCH_CHUNK_SIZE)
                if not rows:
                    break
                for rec_id, blob, payload_json, metadata_json, created_at in rows:
                    vector = _deserialize_vector(blob)
                    score = _cosine_similarity(query.vector, vector)
                    payload: Any | None = json.loads(payload_json) if payload_json else None
                    metadata = json.loads(metadata_json) if metadata_json else None
                    ts = (
                        datetime.fromisoformat(created_at)
                        if isinstance(created_at, str)
                        else datetime.now()
                        if created_at is None
                        else datetime.fromtimestamp(float(created_at))
                    )
                    if query.filter_metadata and metadata:
                        matches = True
                        for key, value in query.filter_metadata.items():
                            if metadata.get(key) != value:
                                matches = False
                                break
                        if not matches:
                            continue

                    record = MemoryRecord(
                        id=rec_id,
                        vector=vector,
                        payload=payload,
                        metadata=metadata,
                        timestamp=ts,
                    )
                    scored = ScoredMemory(record=record, score=score)

                    if len(heap) < limit:
                        heapq.heappush(heap, (score, scored))
                    else:
                        # Maintain min-heap of top-k
                        if heap and score > heap[0][0]:
                            heapq.heapreplace(heap, (score, scored))

            # Sort by score desc then id for deterministic ordering
            heap.sort(key=lambda item: (-item[0], item[1].record.id or ""))
            return [item[1] for item in heap]
        finally:
            conn.close()

    async def delete(self, ids: Sequence[str]) -> None:
        await self._init()
        if not ids:
            return
        await asyncio.to_thread(self._delete_sync, list(ids))

    def _delete_sync(self, ids: list[str]) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()

    async def close(self) -> None:
        # No persistent connection held; nothing to close.
        return None
