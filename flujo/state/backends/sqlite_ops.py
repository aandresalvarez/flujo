"""SQLite-backed persistent storage for workflow state with optimized schema."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, cast

import aiosqlite
import atexit

from flujo.infra import telemetry
from flujo.utils.serialization import robust_serialize, safe_deserialize
from .sqlite_core import SQLiteBackendBase, _fast_json_dumps
from .sqlite_trace import SQLiteTraceMixin


class SQLiteBackend(SQLiteTraceMixin, SQLiteBackendBase):
    async def save_trace(self, run_id: str, trace: Dict[str, Any]) -> None:
        return await SQLiteTraceMixin.save_trace(self, run_id, trace)

    async def get_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        return await SQLiteTraceMixin.get_trace(self, run_id)

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Save workflow state to the database.

        Args:
            run_id: Unique identifier for the workflow run
            state: Dictionary containing workflow state data
        """
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                conn = await self._create_connection()
                try:
                    db = conn
                    # OPTIMIZATION: Use more efficient serialization for performance-critical scenarios
                    # Skip expensive robust_serialize for simple data types
                    pipeline_context = state["pipeline_context"]
                    if isinstance(pipeline_context, dict):
                        # For simple dicts, use direct JSON serialization
                        pipeline_context_json = _fast_json_dumps(pipeline_context)
                    else:
                        # For complex objects, use robust serialization
                        pipeline_context_json = _fast_json_dumps(robust_serialize(pipeline_context))

                    last_step_output = state.get("last_step_output")
                    if last_step_output is not None:
                        if isinstance(last_step_output, (str, int, float, bool, type(None))):
                            # For simple types, use direct JSON serialization
                            last_step_output_json = _fast_json_dumps(last_step_output)
                        else:
                            # For complex objects, use robust serialization
                            last_step_output_json = _fast_json_dumps(
                                robust_serialize(last_step_output)
                            )
                    else:
                        last_step_output_json = None

                    step_history = state.get("step_history")
                    if step_history is not None:
                        if isinstance(step_history, list) and all(
                            isinstance(item, dict) for item in step_history
                        ):
                            # For simple list of dicts, use direct JSON serialization
                            step_history_json = _fast_json_dumps(step_history)
                        else:
                            # For complex objects, use robust serialization
                            step_history_json = _fast_json_dumps(robust_serialize(step_history))
                    else:
                        step_history_json = None

                    # Get execution_time_ms directly from state
                    execution_time_ms = state.get("execution_time_ms")
                    metadata_json = None
                    metadata = state.get("metadata")
                    if metadata is not None:
                        if isinstance(metadata, dict):
                            metadata_json = _fast_json_dumps(metadata)
                        else:
                            metadata_json = _fast_json_dumps(robust_serialize(metadata))

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO workflow_state (
                            run_id, pipeline_id, pipeline_name, pipeline_version,
                            current_step_index, pipeline_context, last_step_output, step_history,
                            status, created_at, updated_at, total_steps,
                            error_message, execution_time_ms, memory_usage_mb,
                            metadata, is_background_task, parent_run_id, task_id, background_error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            state["pipeline_id"],
                            state["pipeline_name"],
                            state["pipeline_version"],
                            state["current_step_index"],
                            pipeline_context_json,
                            last_step_output_json,
                            step_history_json,
                            state["status"],
                            state["created_at"].isoformat(),
                            state["updated_at"].isoformat(),
                            state.get("total_steps", 0),
                            state.get("error_message"),
                            execution_time_ms,
                            state.get("memory_usage_mb"),
                            metadata_json,
                            1 if state.get("is_background_task") else 0,
                            state.get("parent_run_id"),
                            state.get("task_id"),
                            state.get("background_error"),
                        ),
                    )
                    await db.commit()
                    telemetry.logfire.debug(f"Saved state for run_id={run_id}")
                finally:
                    await conn.close()

            await self._with_retries(_save)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:

            async def _load() -> Optional[Dict[str, Any]]:
                conn = await self._create_connection()
                try:
                    db = conn
                    cursor = await db.execute(
                        """
                        SELECT run_id, pipeline_id, pipeline_name, pipeline_version, current_step_index,
                               pipeline_context, last_step_output, step_history, status, created_at, updated_at,
                               total_steps, error_message, execution_time_ms, memory_usage_mb,
                               metadata, is_background_task, parent_run_id, task_id, background_error
                        FROM workflow_state WHERE run_id = ?
                        """,
                        (run_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                finally:
                    await conn.close()
                if row is None:
                    return None
                pipeline_context = (
                    safe_deserialize(json.loads(row[5])) if row[5] is not None else {}
                )
                last_step_output = (
                    safe_deserialize(json.loads(row[6])) if row[6] is not None else None
                )
                step_history = safe_deserialize(json.loads(row[7])) if row[7] is not None else []
                metadata = safe_deserialize(json.loads(row[15])) if row[15] is not None else {}
                return {
                    "run_id": row[0],
                    "pipeline_id": row[1],
                    "pipeline_name": row[2],
                    "pipeline_version": row[3],
                    "current_step_index": row[4],
                    "pipeline_context": pipeline_context,
                    "last_step_output": last_step_output,
                    "step_history": step_history,
                    "status": row[8],
                    "created_at": datetime.fromisoformat(row[9]),
                    "updated_at": datetime.fromisoformat(row[10]),
                    "total_steps": row[11] or 0,
                    "error_message": row[12],
                    "execution_time_ms": row[13],
                    "memory_usage_mb": row[14],
                    "metadata": metadata,
                    "is_background_task": bool(row[16]) if row[16] is not None else False,
                    "parent_run_id": row[17],
                    "task_id": row[18],
                    "background_error": row[19],
                }

            result = await self._with_retries(_load)
            return cast(Optional[Dict[str, Any]], result)

    async def delete_state(self, run_id: str) -> None:
        """Delete workflow state."""
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
                await db.execute("DELETE FROM workflow_state WHERE run_id = ?", (run_id,))
                await db.commit()
            finally:
                await conn.close()

    async def list_states(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List workflow states with optional status filter.

        Uses the pooled connection when available to minimize connection overhead.
        """
        await self._ensure_init()
        async with self._lock:
            db = self._connection_pool
            _temp_conn = False
            if db is None:
                db = await self._create_connection()
                _temp_conn = True
                try:
                    await db.execute("PRAGMA busy_timeout = 1000")
                except Exception:
                    pass
            try:
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
            finally:
                if _temp_conn:
                    try:
                        await db.close()
                    except Exception:
                        pass

    async def list_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List background tasks with optional filtering."""
        await self._ensure_init()
        async with self._lock:
            db = self._connection_pool
            _temp_conn = False
            if db is None:
                db = await self._create_connection()
                _temp_conn = True

            try:
                query = """
                    SELECT run_id, status, created_at, updated_at, metadata, parent_run_id,
                           task_id, background_error
                    FROM workflow_state
                    WHERE is_background_task = 1
                """
                params: List[Any] = []
                if parent_run_id is not None:
                    query += " AND parent_run_id = ?"
                    params.append(parent_run_id)
                if status is not None:
                    query += " AND status = ?"
                    params.append(status)
                query += " ORDER BY updated_at DESC"
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                    query += " OFFSET ?"
                    params.append(offset)

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                tasks: List[Dict[str, Any]] = []
                for row in rows:
                    metadata_raw = safe_deserialize(json.loads(row[4])) if row[4] else {}
                    tasks.append(
                        {
                            "run_id": row[0],
                            "status": row[1],
                            "created_at": row[2],
                            "updated_at": row[3],
                            "metadata": metadata_raw,
                            "parent_run_id": row[5],
                            "task_id": row[6],
                            "background_error": row[7],
                        }
                    )
                return tasks
            finally:
                if _temp_conn:
                    try:
                        await db.close()
                    except Exception:
                        pass

    async def get_failed_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        hours_back: int = 24,
    ) -> List[Dict[str, Any]]:
        """Return failed background tasks within a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        tasks = await self.list_background_tasks(parent_run_id=parent_run_id, status="failed")
        filtered: List[Dict[str, Any]] = []

        def _normalize_ts(value: Any) -> Optional[datetime]:
            dt: Optional[datetime] = None
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    return None
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt

        for task in tasks:
            updated_at = task.get("updated_at")
            updated_dt = _normalize_ts(updated_at)
            if updated_dt is None or updated_dt < cutoff:
                continue
            filtered.append(task)
        return filtered

    async def cleanup_stale_background_tasks(self, stale_hours: int = 24) -> int:
        """Mark running background tasks older than the cutoff as failed."""
        await self._ensure_init()
        async with self._lock:
            db = self._connection_pool
            _temp_conn = False
            if db is None:
                db = await self._create_connection()
                _temp_conn = True

            try:
                cutoff = datetime.now() - timedelta(hours=stale_hours)
                async with db.execute(
                    """
                    SELECT run_id, updated_at FROM workflow_state
                    WHERE is_background_task = 1
                      AND status = 'running'
                      AND updated_at < ?
                    """,
                    (cutoff.isoformat(),),
                ) as cursor:
                    rows = await cursor.fetchall()
                if not rows:
                    return 0

                count = 0
                for run_id, _updated_at in rows:
                    await db.execute(
                        """
                        UPDATE workflow_state
                        SET status = 'failed',
                            background_error = 'Task timeout: process likely crashed',
                            updated_at = ?
                        WHERE run_id = ?
                        """,
                        (datetime.now().isoformat(), run_id),
                    )
                    count += 1
                await db.commit()
                telemetry.logfire.warning(
                    f"Cleaned up {count} stale background tasks (older than {stale_hours} hours)"
                )
                return count
            finally:
                if _temp_conn:
                    try:
                        await db.close()
                    except Exception:
                        pass

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
            # Prefer pooled connection for lower latency; fallback to ad-hoc connection
            db = self._connection_pool
            _temp_conn = False
            if db is None:
                db = await self._create_connection()
                try:
                    await db.execute("PRAGMA busy_timeout = 1000")
                except Exception:
                    pass
                _temp_conn = True
            try:
                # Build query with optional filters
                query = """
                    SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                           current_step_index, status, created_at, updated_at,
                           total_steps, error_message, execution_time_ms, memory_usage_mb,
                           metadata, is_background_task, parent_run_id, task_id, background_error
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
                    metadata = safe_deserialize(json.loads(row[12])) if row[12] else {}
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
                            "metadata": metadata,
                            "is_background_task": bool(row[13]) if row[13] is not None else False,
                            "parent_run_id": row[14],
                            "task_id": row[15],
                            "background_error": row[16],
                        }
                    )
                return result
            finally:
                if _temp_conn:
                    try:
                        await db.close()
                    except Exception:
                        pass

    async def list_runs(
        self,
        status: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List runs from the new structured schema for lens CLI."""
        await self._ensure_init()
        async with self._lock:
            db = self._connection_pool
            _temp_conn = False
            if db is None:
                db = await self._create_connection()
                try:
                    await db.execute("PRAGMA busy_timeout = 1000")
                except Exception:
                    pass
                _temp_conn = True
            try:
                # Optimize query based on filters to use appropriate indexes
                params: List[Any]
                if status and not pipeline_name:
                    # Use status index for better performance
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE status = ?
                        ORDER BY created_at DESC
                    """
                    params = [status]
                elif pipeline_name and not status:
                    # Use pipeline_name index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE pipeline_name = ?
                        ORDER BY created_at DESC
                    """
                    params = [pipeline_name]
                elif status and pipeline_name:
                    # Use both filters
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        WHERE status = ? AND pipeline_name = ?
                        ORDER BY created_at DESC
                    """
                    params = [status, pipeline_name]
                else:
                    # No filters, use created_at index
                    query = """
                        SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at, execution_time_ms
                        FROM runs
                        ORDER BY created_at DESC
                    """
                    params = []

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(int(limit))
                if offset:
                    query += " OFFSET ?"
                    params.append(int(offset))

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                result: List[Dict[str, Any]] = []
                for row in rows:
                    if row is None:
                        continue
                    result.append(
                        {
                            "run_id": row[0],
                            "pipeline_name": row[1],
                            "pipeline_version": row[2],
                            "status": row[3],
                            "start_time": row[
                                4
                            ],  # Map created_at to start_time for backward compatibility
                            "end_time": row[
                                5
                            ],  # Map updated_at to end_time for backward compatibility
                            "total_cost": row[6]
                            if row[6] is not None
                            else 0.0,  # Map execution_time_ms to total_cost for backward compatibility
                        }
                    )
                return result
            finally:
                if _temp_conn:
                    try:
                        await db.close()
                    except Exception:
                        pass

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow statistics."""
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
                # Get total count
                cursor = await db.execute("SELECT COUNT(*) FROM workflow_state")
                total_workflows_row = await cursor.fetchone()
                total_workflows = total_workflows_row[0] if total_workflows_row else 0
                await cursor.close()

                # Get status counts
                cursor = await db.execute(
                    """
                    SELECT status, COUNT(*)
                    FROM workflow_state
                    GROUP BY status
                """
                )
                status_counts_rows = await cursor.fetchall()
                status_counts: Dict[str, int] = {
                    row[0]: row[1] for row in status_counts_rows if row is not None
                }
                await cursor.close()

                # Get recent workflows (last 24 hours)
                cursor = await db.execute(
                    """
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at >= datetime('now', '-24 hours')
                """
                )
                recent_workflows_24h_row = await cursor.fetchone()
                recent_workflows_24h = (
                    recent_workflows_24h_row[0] if recent_workflows_24h_row else 0
                )
                await cursor.close()

                # Get average execution time
                cursor = await db.execute(
                    """
                    SELECT AVG(execution_time_ms)
                    FROM workflow_state
                    WHERE execution_time_ms IS NOT NULL
                """
                )
                avg_exec_time_row = await cursor.fetchone()
                avg_exec_time = avg_exec_time_row[0] if avg_exec_time_row else 0
                await cursor.close()

                # Background task breakdown
                cursor = await db.execute(
                    """
                    SELECT status, COUNT(*) FROM workflow_state
                    WHERE is_background_task = 1
                    GROUP BY status
                    """
                )
                bg_counts_rows = await cursor.fetchall()
                bg_status_counts: Dict[str, int] = {
                    row[0]: row[1] for row in bg_counts_rows if row is not None
                }
                await cursor.close()

                return {
                    "total_workflows": total_workflows,
                    "status_counts": status_counts,
                    "recent_workflows_24h": recent_workflows_24h,
                    "average_execution_time_ms": avg_exec_time or 0,
                    "background_status_counts": bg_status_counts,
                }
            finally:
                await conn.close()

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours."""
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
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
            finally:
                await conn.close()

    async def cleanup_old_workflows(self, days_old: float = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        await self._ensure_init()
        async with self._lock:

            async def _cleanup() -> int:
                conn = await self._create_connection()
                try:
                    db = conn
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
                finally:
                    await conn.close()

            result = await self._with_retries(_cleanup)
            return cast(int, result)

    # ------------------------------------------------------------------
    # New structured persistence API
    # ------------------------------------------------------------------

    async def save_run_start(self, run_data: Dict[str, Any]) -> None:
        async with self._lock:
            # Fast path: if backend not fully initialized yet, bootstrap only the 'runs' table
            # to avoid heavy DDL/PRAGMA costs on first write. Full initialization (other tables,
            # indexes, pragmas) will happen lazily on subsequent calls that need them.
            if not self._initialized:
                try:
                    import sqlite3 as _sqlite

                    created_at = (
                        run_data.get("created_at") or datetime.now(timezone.utc).isoformat()
                    )
                    updated_at = run_data.get("updated_at") or created_at
                    with _sqlite.connect(self.db_path) as _db:
                        # Fast pragmas to reduce fsync/IO cost on first write
                        try:
                            _db.execute("PRAGMA journal_mode = WAL")
                            _db.execute("PRAGMA synchronous = NORMAL")
                            _db.execute("PRAGMA temp_store = MEMORY")
                            _db.execute("PRAGMA cache_size = 10000")
                            _db.execute("PRAGMA mmap_size = 268435456")
                            _db.execute("PRAGMA page_size = 4096")
                            _db.execute("PRAGMA busy_timeout = 1000")
                        except Exception:
                            pass
                        _db.execute(
                            """
                            CREATE TABLE IF NOT EXISTS runs (
                                run_id TEXT PRIMARY KEY,
                                pipeline_id TEXT NOT NULL,
                                pipeline_name TEXT NOT NULL,
                                pipeline_version TEXT NOT NULL,
                                status TEXT NOT NULL,
                                created_at TEXT NOT NULL,
                                updated_at TEXT NOT NULL
                            )
                            """
                        )
                        _db.execute(
                            """
                            INSERT OR REPLACE INTO runs (
                                run_id, pipeline_id, pipeline_name, pipeline_version, status,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_data["run_id"],
                                run_data.get("pipeline_id", "unknown"),
                                run_data.get("pipeline_name", "unknown"),
                                run_data.get("pipeline_version", "latest"),
                                run_data.get("status", "running"),
                                created_at,
                                updated_at,
                            ),
                        )
                        _db.commit()
                    return
                except Exception:
                    # Fall back to full initialization path on any error
                    pass

            await self._ensure_init()

            async def _save() -> None:
                # Use pooled connection when available to avoid connect() overhead
                db = self._connection_pool
                temp_conn: Optional[aiosqlite.Connection] = None
                if db is None:
                    temp_conn = await self._create_connection()
                    db = temp_conn

                try:
                    # OPTIMIZATION: Use simplified schema for better performance
                    created_at = (
                        run_data.get("created_at") or datetime.now(timezone.utc).isoformat()
                    )
                    updated_at = run_data.get("updated_at") or created_at

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO runs (
                            run_id, pipeline_id, pipeline_name, pipeline_version, status,
                            created_at, updated_at, execution_time_ms, memory_usage_mb,
                            total_steps, error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_data["run_id"],
                            run_data.get("pipeline_id", "unknown"),
                            run_data.get("pipeline_name", "unknown"),
                            run_data.get("pipeline_version", "latest"),
                            run_data.get("status", "running"),
                            created_at,
                            updated_at,
                            run_data.get("execution_time_ms"),
                            run_data.get("memory_usage_mb"),
                            run_data.get("total_steps", 0),
                            run_data.get("error_message"),
                        ),
                    )
                    await db.commit()
                finally:
                    if temp_conn is not None:
                        await temp_conn.close()

            await self._with_retries(_save)

    async def save_step_result(self, step_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                # Prefer pooled connection to avoid connection setup overhead on hot paths
                db = self._connection_pool
                temp_conn: Optional[aiosqlite.Connection] = None
                if db is None:
                    temp_conn = await self._create_connection()
                    db = temp_conn
                try:
                    await db.execute("PRAGMA foreign_keys = ON")
                    await db.execute("PRAGMA busy_timeout = 1000")
                except Exception:
                    pass

                try:
                    # Ensure parent run row exists to avoid FK failures in concurrent/resume scenarios
                    try:
                        run_id = step_data.get("run_id")
                        if run_id is not None:
                            now = datetime.now(timezone.utc).isoformat()
                            await db.execute(
                                """
                                INSERT OR IGNORE INTO runs (
                                    run_id, pipeline_id, pipeline_name, pipeline_version, status,
                                    created_at, updated_at
                                ) VALUES (?, 'unknown', 'unknown', 'latest', 'running', ?, ?)
                                """,
                                (run_id, now, now),
                            )
                            # Do not commit yet; the final commit below will include both insert and step
                    except sqlite3.Error:
                        # Best-effort only; continue to step insert and let FK enforcement handle it
                        pass

                    # OPTIMIZATION: Use simplified schema for better performance
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO steps (
                            run_id, step_name, step_index, status, output, raw_response, cost_usd,
                            token_counts, execution_time_ms, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            step_data["run_id"],
                            step_data["step_name"],
                            step_data["step_index"],
                            step_data.get("status", "completed"),
                            _fast_json_dumps(step_data.get("output")),
                            _fast_json_dumps(step_data.get("raw_response"))
                            if step_data.get("raw_response") is not None
                            else None,
                            step_data.get("cost_usd"),
                            step_data.get("token_counts"),
                            step_data.get("execution_time_ms"),
                            step_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                        ),
                    )
                    await db.commit()
                finally:
                    if temp_conn is not None:
                        await temp_conn.close()

            await self._with_retries(_save)

    async def save_run_end(self, run_id: str, end_data: Dict[str, Any]) -> None:
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                db = self._connection_pool
                temp_conn: Optional[aiosqlite.Connection] = None
                if db is None:
                    temp_conn = await self._create_connection()
                    db = temp_conn
                try:
                    await db.execute("PRAGMA busy_timeout = 1000")
                except Exception:
                    pass

                try:
                    await db.execute(
                        """
                        UPDATE runs
                        SET status = ?, updated_at = ?, execution_time_ms = ?,
                            memory_usage_mb = ?, total_steps = ?, error_message = ?
                        WHERE run_id = ?
                        """,
                        (
                            end_data.get("status", "completed"),
                            end_data.get("updated_at", datetime.now(timezone.utc).isoformat()),
                            end_data.get("execution_time_ms"),
                            end_data.get("memory_usage_mb"),
                            end_data.get("total_steps", 0),
                            end_data.get("error_message"),
                            run_id,
                        ),
                    )
                    await db.commit()
                finally:
                    if temp_conn is not None:
                        await temp_conn.close()

            await self._with_retries(_save)

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
                cursor = await db.execute(
                    """
                    SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at,
                           execution_time_ms, memory_usage_mb, total_steps, error_message
                    FROM runs WHERE run_id = ?
                    """,
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    return None
                return {
                    "run_id": row[0],
                    "pipeline_name": row[1],
                    "pipeline_version": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "execution_time_ms": row[6],
                    "memory_usage_mb": row[7],
                    "total_steps": row[8],
                    "error_message": row[9],
                }
            finally:
                await conn.close()

    async def list_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
                cursor = await db.execute(
                    """
                    SELECT step_name, step_index, status, output, raw_response, cost_usd, token_counts,
                           execution_time_ms, created_at
                    FROM steps WHERE run_id = ? ORDER BY step_index
                    """,
                    (run_id,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                results: List[Dict[str, Any]] = []
                for r in rows:
                    results.append(
                        {
                            "step_name": r[0],
                            "step_index": r[1],
                            "status": r[2],
                            "output": safe_deserialize(json.loads(r[3])) if r[3] else None,
                            "raw_response": safe_deserialize(json.loads(r[4])) if r[4] else None,
                            "cost_usd": r[5],
                            "token_counts": r[6],
                            "execution_time_ms": r[7],
                            "created_at": r[8],
                        }
                    )
                return results
            finally:
                await conn.close()

    async def delete_run(self, run_id: str) -> None:
        """Delete a run from the runs table (cascades to traces). Audit log deletion."""
        await self._ensure_init()
        async with self._lock:
            try:
                from flujo.infra.audit import log_audit as _audit

                _audit("run_deleted", run_id=run_id)
            except Exception:
                pass
            conn = await self._create_connection()
            try:
                db = conn
                await db.execute("PRAGMA foreign_keys = ON")
                await db.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                await db.commit()
            finally:
                await conn.close()


# Ensure we shut down any pooled connections on interpreter exit
try:
    atexit.register(SQLiteBackend.shutdown_all)
except Exception:
    pass
