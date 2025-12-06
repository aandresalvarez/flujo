from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
from pathlib import Path
from typing import Sequence, Any
from datetime import datetime

from ...domain.memory import (
    MemoryRecord,
    ScoredMemory,
    VectorQuery,
    VectorStoreProtocol,
    _assign_id,
    _cosine_similarity,
)


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
        rows = await asyncio.to_thread(self._fetch_all)
        results: list[ScoredMemory] = []
        for row in rows:
            rec_id, blob, payload_json, metadata_json, created_at = row
            vector = _deserialize_vector(blob)
            score = _cosine_similarity(query.vector, vector)
            payload: Any | None = json.loads(payload_json) if payload_json else None
            metadata = json.loads(metadata_json) if metadata_json else None
            if isinstance(created_at, str):
                ts = datetime.fromisoformat(created_at)
            elif created_at is None:
                ts = datetime.now()
            else:
                ts = datetime.fromtimestamp(float(created_at))
            record = MemoryRecord(
                id=rec_id,
                vector=vector,
                payload=payload,
                metadata=metadata,
                timestamp=ts,
            )
            if query.filter_metadata and metadata:
                matches = True
                for key, value in query.filter_metadata.items():
                    if metadata.get(key) != value:
                        matches = False
                        break
                if not matches:
                    continue
            results.append(ScoredMemory(record=record, score=score))
        results.sort(key=lambda item: (-item.score, item.record.id or ""))
        return results[: max(query.limit, 0)]

    def _fetch_all(self) -> list[tuple[str, bytes, str | None, str | None, str | None]]:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.execute("SELECT id, vector, payload, metadata, created_at FROM memories")
            return list(cur.fetchall())
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
