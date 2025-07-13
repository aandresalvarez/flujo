from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Any, Dict, List, Optional, cast

import aiosqlite
import sqlite3

from .base import StateBackend
from ...utils.serialization import safe_serialize, safe_deserialize
from ...infra import telemetry


class SQLiteBackend(StateBackend):
    """SQLite-backed persistent storage for workflow state with optimized schema."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False
        # Remove self._logger

    async def _init_db(self) -> None:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_state (
                        run_id TEXT PRIMARY KEY,
                        pipeline_id TEXT NOT NULL,
                        pipeline_name TEXT NOT NULL,
                        pipeline_version TEXT NOT NULL,
                        current_step_index INTEGER NOT NULL DEFAULT 0,
                        pipeline_context TEXT NOT NULL,
                        last_step_output TEXT,
                        status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'failed', 'cancelled')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        total_steps INTEGER DEFAULT 0,
                        error_message TEXT,
                        execution_time_ms INTEGER,
                        memory_usage_mb REAL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
                )
                await self._migrate_existing_schema(db)
                await db.commit()
            telemetry.logfire.info(f"Initialized SQLite database at {self.db_path}")
        except sqlite3.DatabaseError as e:
            telemetry.logfire.error(
                f"Database corruption detected: {e}. Reinitializing {self.db_path}."
            )
            backup_path = self.db_path.with_suffix(self.db_path.suffix + ".corrupt")
            self.db_path.rename(backup_path)
            telemetry.logfire.warn(f"Corrupted DB moved to {backup_path}")
            await self._init_db()

    async def _migrate_existing_schema(self, db: aiosqlite.Connection) -> None:
        """Migrate existing database schema to the new optimized structure."""
        cursor = await db.execute("PRAGMA table_info(workflow_state)")
        existing_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        # Add new columns if they don't exist
        new_columns = [
            ("total_steps", "INTEGER DEFAULT 0"),
            ("error_message", "TEXT"),
            ("execution_time_ms", "INTEGER"),
            ("memory_usage_mb", "REAL"),
        ]

        for column_name, column_def in new_columns:
            if column_name not in existing_columns:
                await db.execute(
                    f"ALTER TABLE workflow_state ADD COLUMN {column_name} {column_def}"
                )

        # Ensure required columns exist with proper constraints
        if "pipeline_name" not in existing_columns:
            await db.execute(
                "ALTER TABLE workflow_state ADD COLUMN pipeline_name TEXT NOT NULL DEFAULT ''"
            )
            await db.execute(
                "UPDATE workflow_state SET pipeline_name = pipeline_id WHERE pipeline_name = ''"
            )

        # Update any NULL values in required columns
        await db.execute(
            "UPDATE workflow_state SET current_step_index = 0 WHERE current_step_index IS NULL"
        )
        await db.execute(
            "UPDATE workflow_state SET pipeline_context = '{}' WHERE pipeline_context IS NULL"
        )
        await db.execute("UPDATE workflow_state SET status = 'running' WHERE status IS NULL")

    async def _ensure_init(self) -> None:
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    try:
                        await self._init_db()
                        self._initialized = True
                    except sqlite3.DatabaseError as e:
                        telemetry.logfire.error(f"Failed to initialize DB: {e}")
                        raise

    async def _with_retries(
        self, coro_func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        retries = 3
        delay = 0.1
        for attempt in range(retries):
            try:
                result = await coro_func(*args, **kwargs)
                return result
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < retries - 1:
                    telemetry.logfire.warn(
                        f"Database is locked, retrying ({attempt + 1}/{retries})..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
            except sqlite3.DatabaseError as e:
                if "no such column" in str(e).lower():
                    telemetry.logfire.warn(f"Schema mismatch detected: {e}. Attempting migration.")
                    await self._init_db()
                    continue
                raise

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                async with aiosqlite.connect(self.db_path) as db:
                    execution_time_ms = None
                    if state.get("execution_time_ms") is not None:
                        execution_time_ms = state["execution_time_ms"]
                    pipeline_context_json = json.dumps(safe_serialize(state["pipeline_context"]))
                    last_step_output_json = (
                        json.dumps(safe_serialize(state["last_step_output"]))
                        if state.get("last_step_output") is not None
                        else None
                    )
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO workflow_state (
                            run_id, pipeline_id, pipeline_name, pipeline_version,
                            current_step_index, pipeline_context, last_step_output,
                            status, created_at, updated_at, total_steps,
                            error_message, execution_time_ms, memory_usage_mb
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            state["pipeline_id"],
                            state["pipeline_name"],
                            state["pipeline_version"],
                            state["current_step_index"],
                            pipeline_context_json,
                            last_step_output_json,
                            state["status"],
                            state["created_at"].isoformat(),
                            state["updated_at"].isoformat(),
                            state.get("total_steps", 0),
                            state.get("error_message"),
                            execution_time_ms,
                            state.get("memory_usage_mb"),
                        ),
                    )
                    await db.commit()
                    telemetry.logfire.info(f"Saved state for run_id={run_id}")

            await self._with_retries(_save)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:

            async def _load() -> Optional[Dict[str, Any]]:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        """
                        SELECT run_id, pipeline_id, pipeline_name, pipeline_version, current_step_index,
                               pipeline_context, last_step_output, status, created_at, updated_at,
                               total_steps, error_message, execution_time_ms, memory_usage_mb
                        FROM workflow_state WHERE run_id = ?
                        """,
                        (run_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                if row is None:
                    return None
                pipeline_context = (
                    safe_deserialize(json.loads(row[5])) if row[5] is not None else {}
                )
                last_step_output = (
                    safe_deserialize(json.loads(row[6])) if row[6] is not None else None
                )
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
                    "total_steps": row[10] or 0,
                    "error_message": row[11],
                    "execution_time_ms": row[12],
                    "memory_usage_mb": row[13],
                }

            result = await self._with_retries(_load)
            return cast(Optional[Dict[str, Any]], result)

    async def delete_state(self, run_id: str) -> None:
        """Delete workflow state."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM workflow_state WHERE run_id = ?", (run_id,))
                await db.commit()

    async def list_states(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow states with optional status filter."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                if status:
                    async with db.execute(
                        "SELECT run_id, status, created_at, updated_at FROM workflow_state WHERE status = ? ORDER BY created_at DESC",
                        (status,),
                    ) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with db.execute(
                        "SELECT run_id, status, created_at, updated_at FROM workflow_state ORDER BY created_at DESC"
                    ) as cursor:
                        rows = await cursor.fetchall()

                return [
                    {
                        "run_id": row[0],
                        "status": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                    }
                    for row in rows
                ]

    async def list_workflows(
        self,
        status: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Enhanced workflow listing with additional filters and metadata."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query with optional filters
                query = """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                           current_step_index, status, created_at, updated_at,
                           total_steps, error_message, execution_time_ms, memory_usage_mb
                    FROM workflow_state
                    WHERE 1=1
                """
                params: List[Any] = []

                if status:
                    query += " AND status = ?"
                    params.append(status)

                if pipeline_id:
                    query += " AND pipeline_id = ?"
                    params.append(pipeline_id)

                query += " ORDER BY created_at DESC"

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_id": row[1],
                            "pipeline_name": row[2],
                            "pipeline_version": row[3],
                            "current_step_index": row[4],
                            "status": row[5],
                            "created_at": row[6],
                            "updated_at": row[7],
                            "total_steps": row[8] or 0,
                            "error_message": row[9],
                            "execution_time_ms": row[10],
                            "memory_usage_mb": row[11],
                        }
                    )
                return result

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Get total count
                cursor = await db.execute("SELECT COUNT(*) FROM workflow_state")
                total_workflows_row = await cursor.fetchone()
                total_workflows = total_workflows_row[0] if total_workflows_row else 0
                await cursor.close()

                # Get status counts
                cursor = await db.execute("""
                    SELECT status, COUNT(*)
                    FROM workflow_state
                    GROUP BY status
                """)
                status_counts_rows = await cursor.fetchall()
                status_counts: Dict[str, int] = {
                    row[0]: row[1] for row in status_counts_rows if row is not None
                }
                await cursor.close()

                # Get recent workflows (last 24 hours)
                cursor = await db.execute("""
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at >= datetime('now', '-24 hours')
                """)
                recent_workflows_24h_row = await cursor.fetchone()
                recent_workflows_24h = (
                    recent_workflows_24h_row[0] if recent_workflows_24h_row else 0
                )
                await cursor.close()

                # Get average execution time
                cursor = await db.execute("""
                    SELECT AVG(execution_time_ms)
                    FROM workflow_state
                    WHERE execution_time_ms IS NOT NULL
                """)
                avg_exec_time_row = await cursor.fetchone()
                avg_exec_time = avg_exec_time_row[0] if avg_exec_time_row else 0
                await cursor.close()

                return {
                    "total_workflows": total_workflows,
                    "status_counts": status_counts,
                    "recent_workflows_24h": recent_workflows_24h,
                    "average_execution_time_ms": avg_exec_time or 0,
                }

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours."""
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                           current_step_index, status, created_at, updated_at,
                           total_steps, error_message, execution_time_ms, memory_usage_mb
                    FROM workflow_state
                    WHERE status = 'failed'
                    AND updated_at >= datetime('now', '-' || ? || ' hours')
                    ORDER BY updated_at DESC
                """,
                    (hours_back,),
                )

                rows = await cursor.fetchall()
                await cursor.close()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_id": row[1],
                            "pipeline_name": row[2],
                            "pipeline_version": row[3],
                            "current_step_index": row[4],
                            "status": row[5],
                            "created_at": row[6],
                            "updated_at": row[7],
                            "total_steps": row[8] or 0,
                            "error_message": row[9],
                            "execution_time_ms": row[10],
                            "memory_usage_mb": row[11],
                        }
                    )
                return result

    async def cleanup_old_workflows(self, days_old: float = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        await self._ensure_init()
        async with self._lock:

            async def _cleanup() -> int:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute(
                        """
                        SELECT COUNT(*)
                        FROM workflow_state
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                        """,
                        (days_old,),
                    )
                    count_row = await cursor.fetchone()
                    count = count_row[0] if count_row else 0
                    await cursor.close()
                    await db.execute(
                        """
                        DELETE FROM workflow_state
                        WHERE created_at < datetime('now', '-' || ? || ' days')
                        """,
                        (days_old,),
                    )
                    await db.commit()
                    telemetry.logfire.info(
                        f"Cleaned up {count} old workflows older than {days_old} days"
                    )
                    return int(count)

            result = await self._with_retries(_cleanup)
            return cast(int, result)
