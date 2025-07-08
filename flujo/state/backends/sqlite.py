from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import orjson
import aiosqlite

from .base import StateBackend


class SQLiteBackend(StateBackend):
    """SQLite-backed persistent storage for workflow state with optimized schema."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _init_db(self) -> None:
        """Initialize database with optimized schema and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable foreign keys and WAL mode for better performance
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("PRAGMA journal_mode = WAL")

            # Create the main workflow state table with optimized schema
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_state (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    pipeline_name TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    current_step_index INTEGER NOT NULL DEFAULT 0,
                    pipeline_context TEXT NOT NULL,  -- JSON blob
                    last_step_output TEXT,           -- JSON blob, nullable
                    status TEXT NOT NULL CHECK (status IN ('running', 'paused', 'completed', 'failed', 'cancelled')),
                    created_at TEXT NOT NULL,        -- ISO format datetime
                    updated_at TEXT NOT NULL,        -- ISO format datetime

                    -- Additional metadata for better observability
                    total_steps INTEGER DEFAULT 0,
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    memory_usage_mb REAL
                )
                """
            )

            # Create indexes for common query patterns
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_updated_at ON workflow_state(updated_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_status_updated ON workflow_state(status, updated_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_status ON workflow_state(pipeline_id, status)"
            )

            # Run migration for existing databases
            await self._migrate_existing_schema(db)

            await db.commit()

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
                    await self._init_db()
                    self._initialized = True

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Calculate execution time if we have previous state
                execution_time_ms = None
                if state.get("execution_time_ms") is not None:
                    execution_time_ms = state["execution_time_ms"]

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
                        updated_at,
                        total_steps,
                        error_message,
                        execution_time_ms,
                        memory_usage_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        state.get("total_steps", 0),
                        state.get("error_message"),
                        execution_time_ms,
                        state.get("memory_usage_mb"),
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
            "total_steps": row[10] or 0,
            "error_message": row[11],
            "execution_time_ms": row[12],
            "memory_usage_mb": row[13],
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

    # New query methods for better observability
    async def list_workflows(
        self,
        status: Optional[Literal["running", "paused", "completed", "failed", "cancelled"]] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filtering and pagination."""
        await self._ensure_init()

        query = "SELECT run_id, pipeline_id, pipeline_name, status, created_at, updated_at, current_step_index FROM workflow_state"
        params = []
        conditions = []

        if status is not None:
            conditions.append("status = ?")
            params.append(str(status))

        if pipeline_id is not None:
            conditions.append("pipeline_id = ?")
            params.append(pipeline_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC"

        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()

        return [
            {
                "run_id": row[0],
                "pipeline_id": row[1],
                "pipeline_name": row[2],
                "status": row[3],
                "created_at": datetime.fromisoformat(row[4]),
                "updated_at": datetime.fromisoformat(row[5]),
                "current_step_index": row[6],
            }
            for row in rows
        ]

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about stored workflows."""
        await self._ensure_init()

        async with aiosqlite.connect(self.db_path) as db:
            # Count by status
            cursor = await db.execute("SELECT status, COUNT(*) FROM workflow_state GROUP BY status")
            status_counts: Dict[str, int] = dict(await cursor.fetchall())  # type: ignore[arg-type]
            await cursor.close()

            # Total count
            cursor = await db.execute("SELECT COUNT(*) FROM workflow_state")
            row = await cursor.fetchone()
            total_count = row[0] if row is not None else 0
            await cursor.close()

            # Recent activity (last 24 hours)
            cursor = await db.execute(
                "SELECT COUNT(*) FROM workflow_state WHERE updated_at > datetime('now', '-1 day')"
            )
            row = await cursor.fetchone()
            recent_count = row[0] if row is not None else 0
            await cursor.close()

            # Average execution time
            cursor = await db.execute(
                "SELECT AVG(execution_time_ms) FROM workflow_state WHERE execution_time_ms IS NOT NULL"
            )
            row = await cursor.fetchone()
            avg_execution_time = row[0] if row is not None else None
            await cursor.close()

        return {
            "total_workflows": total_count,
            "recent_workflows_24h": recent_count,
            "status_counts": status_counts,
            "average_execution_time_ms": avg_execution_time,
        }

    async def cleanup_old_workflows(self, days_old: int = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        await self._ensure_init()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM workflow_state WHERE updated_at < datetime('now', '-{} days')".format(
                    days_old
                )
            )
            deleted_count = cursor.rowcount
            await db.commit()
            await cursor.close()

        return deleted_count

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours with error details."""
        await self._ensure_init()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT run_id, pipeline_id, pipeline_name, error_message, created_at, updated_at
                FROM workflow_state
                WHERE status = 'failed' AND updated_at > datetime('now', '-{} hours')
                ORDER BY updated_at DESC
                """.format(hours_back)
            )
            rows = await cursor.fetchall()
            await cursor.close()

        return [
            {
                "run_id": row[0],
                "pipeline_id": row[1],
                "pipeline_name": row[2],
                "error_message": row[3],
                "created_at": datetime.fromisoformat(row[4]),
                "updated_at": datetime.fromisoformat(row[5]),
            }
            for row in rows
        ]
