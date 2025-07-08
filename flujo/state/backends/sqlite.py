from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
import aiosqlite

from .base import StateBackend


class SQLiteBackend(StateBackend):
    """SQLite-backed persistent storage for workflow state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_state (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT,
                    pipeline_name TEXT,
                    pipeline_version TEXT,
                    current_step_index INTEGER,
                    pipeline_context TEXT,
                    last_step_output TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

            cursor = await db.execute("PRAGMA table_info(workflow_state)")
            cols = [row[1] for row in await cursor.fetchall()]
            await cursor.close()
            if "pipeline_name" not in cols:
                await db.execute("ALTER TABLE workflow_state ADD COLUMN pipeline_name TEXT")
                await db.execute(
                    "UPDATE workflow_state SET pipeline_name = '' WHERE pipeline_name IS NULL"
                )

            await db.commit()

    async def _ensure_init(self) -> None:
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    await self._init_db()
                    self._initialized = True

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO workflow_state (
                        run_id,
                        pipeline_id,
                        pipeline_name,
                        pipeline_version,
                        current_step_index,
                        pipeline_context,
                        last_step_output,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        state["pipeline_id"],
                        state["pipeline_name"],
                        state["pipeline_version"],
                        state["current_step_index"],
                        orjson.dumps(state["pipeline_context"]).decode(),
                        (
                            orjson.dumps(state["last_step_output"]).decode()
                            if state.get("last_step_output") is not None
                            else None
                        ),
                        state["status"],
                        state["created_at"].isoformat(),
                        state["updated_at"].isoformat(),
                    ),
                )
                await db.commit()

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version, current_step_index,
                           pipeline_context, last_step_output, status, created_at,
                           updated_at
                    FROM workflow_state WHERE run_id = ?
                    """,
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
        if row is None:
            return None
        pipeline_context = orjson.loads(row[5]) if row[5] is not None else {}
        last_step_output = orjson.loads(row[6]) if row[6] is not None else None
        return {
            "run_id": row[0],
            "pipeline_id": row[1],
            "pipeline_name": row[2],
            "pipeline_version": row[3],
            "current_step_index": row[4],
            "pipeline_context": pipeline_context,
            "last_step_output": last_step_output,
            "status": row[7],
            "created_at": datetime.fromisoformat(row[8]),
            "updated_at": datetime.fromisoformat(row[9]),
        }

    async def delete_state(self, run_id: str) -> None:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM workflow_state WHERE run_id = ?",
                    (run_id,),
                )
                await db.commit()
