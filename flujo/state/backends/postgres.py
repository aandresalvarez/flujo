from __future__ import annotations

import asyncio
import json
import importlib
import importlib.util
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, cast

from flujo.state.backends.base import StateBackend
from flujo.type_definitions.common import JSONObject
from flujo.utils.serialization import safe_deserialize, safe_serialize

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg
    from asyncpg import Pool, Record
else:  # pragma: no cover - runtime checked import
    asyncpg = None
    Pool = Any
    Record = Any


def _load_asyncpg() -> Any:
    spec = importlib.util.find_spec("asyncpg")
    if spec is None:
        raise RuntimeError("asyncpg is required. Install with `pip install flujo[postgres]`.")
    module = importlib.import_module("asyncpg")
    return module


def _jsonb(value: Any) -> Optional[str]:
    if value is None:
        return None
    serialized = safe_serialize(value)
    return json.dumps(serialized)


class PostgresBackend(StateBackend):
    def __init__(
        self,
        dsn: str,
        *,
        auto_migrate: bool = True,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._auto_migrate = auto_migrate
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: Optional[Pool] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_pool(self) -> Pool:
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._pool is None:
                pg = _load_asyncpg()
                pool = await pg.create_pool(
                    self._dsn, min_size=self._pool_min_size, max_size=self._pool_max_size
                )
                self._pool = cast("Pool", pool)
        return cast("Pool", self._pool)

    async def _ensure_init(self) -> None:
        await self._ensure_pool()
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            pool = cast("Pool", self._pool)
            if self._auto_migrate:
                await self._init_schema(pool)
            else:
                await self._verify_schema(pool)
            self._initialized = True

    async def _verify_schema(self, pool: Pool) -> None:
        async with pool.acquire() as conn:
            exists = await conn.fetchval("SELECT to_regclass('workflow_state')")
            if exists is None:
                raise RuntimeError(
                    "workflow_state table not found; run `flujo migrate` or enable FLUJO_AUTO_MIGRATE"
                )

    async def _init_schema(self, pool: Pool) -> None:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_state (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    pipeline_name TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    current_step_index INTEGER NOT NULL,
                    pipeline_context JSONB,
                    last_step_output JSONB,
                    step_history JSONB,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    total_steps INTEGER DEFAULT 0,
                    error_message TEXT,
                    execution_time_ms INTEGER,
                    memory_usage_mb REAL,
                    metadata JSONB,
                    is_background_task BOOLEAN DEFAULT FALSE,
                    parent_run_id TEXT,
                    task_id TEXT,
                    background_error TEXT
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_id TEXT NOT NULL,
                    pipeline_name TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    execution_time_ms INTEGER,
                    memory_usage_mb REAL,
                    total_steps INTEGER DEFAULT 0,
                    error_message TEXT
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    output JSONB,
                    raw_response JSONB,
                    cost_usd REAL,
                    token_counts INTEGER,
                    execution_time_ms INTEGER,
                    created_at TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    run_id TEXT PRIMARY KEY,
                    trace_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    start_time DOUBLE PRECISION NOT NULL,
                    end_time DOUBLE PRECISION,
                    status TEXT NOT NULL,
                    attributes JSONB,
                    created_at TIMESTAMPTZ NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flujo_schema_versions (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_status ON workflow_state(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_pipeline_id ON workflow_state(pipeline_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_parent_run_id ON workflow_state(parent_run_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_state_created_at ON workflow_state(created_at)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_pipeline_id ON runs(pipeline_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_pipeline_name ON runs(pipeline_name)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_steps_step_index ON steps(step_index)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_run_id ON spans(run_id)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_spans_parent_span ON spans(parent_span_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_context_gin ON workflow_state USING GIN(pipeline_context)"
            )

    async def save_state(self, run_id: str, state: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            created_at = state.get("created_at") or datetime.now(timezone.utc)
            updated_at = state.get("updated_at") or datetime.now(timezone.utc)
            await conn.execute(
                """
                INSERT INTO workflow_state (
                    run_id, pipeline_id, pipeline_name, pipeline_version,
                    current_step_index, pipeline_context, last_step_output, step_history,
                    status, created_at, updated_at, total_steps, error_message,
                    execution_time_ms, memory_usage_mb, metadata, is_background_task,
                    parent_run_id, task_id, background_error
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9,
                    $10, $11, $12, $13, $14, $15, $16::jsonb, $17, $18, $19, $20
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    pipeline_id = EXCLUDED.pipeline_id,
                    pipeline_name = EXCLUDED.pipeline_name,
                    pipeline_version = EXCLUDED.pipeline_version,
                    current_step_index = EXCLUDED.current_step_index,
                    pipeline_context = EXCLUDED.pipeline_context,
                    last_step_output = EXCLUDED.last_step_output,
                    step_history = EXCLUDED.step_history,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    total_steps = EXCLUDED.total_steps,
                    error_message = EXCLUDED.error_message,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    memory_usage_mb = EXCLUDED.memory_usage_mb,
                    metadata = EXCLUDED.metadata,
                    is_background_task = EXCLUDED.is_background_task,
                    parent_run_id = EXCLUDED.parent_run_id,
                    task_id = EXCLUDED.task_id,
                    background_error = EXCLUDED.background_error
                """,
                run_id,
                state["pipeline_id"],
                state["pipeline_name"],
                state["pipeline_version"],
                state["current_step_index"],
                _jsonb(state.get("pipeline_context")),
                _jsonb(state.get("last_step_output")),
                _jsonb(state.get("step_history")),
                state.get("status", "running"),
                created_at,
                updated_at,
                state.get("total_steps", 0),
                state.get("error_message"),
                state.get("execution_time_ms"),
                state.get("memory_usage_mb"),
                _jsonb(state.get("metadata")),
                bool(state.get("is_background_task", False)),
                state.get("parent_run_id"),
                state.get("task_id"),
                state.get("background_error"),
            )

    async def load_state(self, run_id: str) -> Optional[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                SELECT run_id, pipeline_id, pipeline_name, pipeline_version, current_step_index,
                       pipeline_context, last_step_output, step_history, status, created_at,
                       updated_at, total_steps, error_message, execution_time_ms,
                       memory_usage_mb, metadata, is_background_task, parent_run_id, task_id,
                       background_error
                FROM workflow_state WHERE run_id = $1
                """,
                run_id,
            )
            if record is None:
                return None
            return {
                "run_id": record["run_id"],
                "pipeline_id": record["pipeline_id"],
                "pipeline_name": record["pipeline_name"],
                "pipeline_version": record["pipeline_version"],
                "current_step_index": record["current_step_index"],
                "pipeline_context": safe_deserialize(record["pipeline_context"]) or {},
                "last_step_output": safe_deserialize(record["last_step_output"]),
                "step_history": safe_deserialize(record["step_history"]) or [],
                "status": record["status"],
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
                "total_steps": record.get("total_steps", 0),
                "error_message": record["error_message"],
                "execution_time_ms": record["execution_time_ms"],
                "memory_usage_mb": record["memory_usage_mb"],
                "metadata": safe_deserialize(record["metadata"]) or {},
                "is_background_task": bool(record["is_background_task"])
                if record["is_background_task"] is not None
                else False,
                "parent_run_id": record["parent_run_id"],
                "task_id": record["task_id"],
                "background_error": record["background_error"],
            }

    async def delete_state(self, run_id: str) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM workflow_state WHERE run_id = $1", run_id)

    async def get_trace(self, run_id: str) -> Optional[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow("SELECT trace_data FROM traces WHERE run_id = $1", run_id)
            if record is None:
                return None
            return cast(JSONObject, safe_deserialize(record["trace_data"]))

    async def save_trace(self, run_id: str, trace: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            now = datetime.now(timezone.utc)
            await conn.execute(
                """
                INSERT INTO traces (run_id, trace_data, created_at)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (run_id) DO UPDATE SET trace_data = EXCLUDED.trace_data,
                    created_at = EXCLUDED.created_at
                """,
                run_id,
                _jsonb(trace),
                now,
            )

    async def save_run_start(self, run_data: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs (
                    run_id, pipeline_id, pipeline_name, pipeline_version, status,
                    created_at, updated_at, execution_time_ms, memory_usage_mb, total_steps,
                    error_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                ) ON CONFLICT (run_id) DO NOTHING
                """,
                run_data["run_id"],
                run_data["pipeline_id"],
                run_data.get("pipeline_name", run_data.get("pipeline_id")),
                run_data.get("pipeline_version", "1.0"),
                run_data.get("status", "running"),
                run_data.get("created_at", datetime.now(timezone.utc)),
                run_data.get("updated_at", datetime.now(timezone.utc)),
                run_data.get("execution_time_ms"),
                run_data.get("memory_usage_mb"),
                run_data.get("total_steps", 0),
                run_data.get("error_message"),
            )

    async def save_step_result(self, step_data: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO steps (
                    run_id, step_name, step_index, status, output, raw_response, cost_usd,
                    token_counts, execution_time_ms, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10
                )
                """,
                step_data["run_id"],
                step_data["step_name"],
                step_data["step_index"],
                step_data.get("status", "completed"),
                _jsonb(step_data.get("output")),
                _jsonb(step_data.get("raw_response")),
                step_data.get("cost_usd"),
                step_data.get("token_counts"),
                step_data.get("execution_time_ms"),
                step_data.get("created_at", datetime.now(timezone.utc)),
            )

    async def save_run_end(self, run_id: str, end_data: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runs
                SET status = $1, updated_at = $2, execution_time_ms = $3,
                    memory_usage_mb = $4, total_steps = $5, error_message = $6
                WHERE run_id = $7
                """,
                end_data.get("status", "completed"),
                end_data.get("updated_at", datetime.now(timezone.utc)),
                end_data.get("execution_time_ms"),
                end_data.get("memory_usage_mb"),
                end_data.get("total_steps", 0),
                end_data.get("error_message"),
                run_id,
            )

    async def get_run_details(self, run_id: str) -> Optional[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT run_id, pipeline_name, pipeline_version, status, created_at, updated_at,
                       execution_time_ms, memory_usage_mb, total_steps, error_message
                FROM runs WHERE run_id = $1
                """,
                run_id,
            )
            if row is None:
                return None
            return {
                "run_id": row["run_id"],
                "pipeline_name": row["pipeline_name"],
                "pipeline_version": row["pipeline_version"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "execution_time_ms": row["execution_time_ms"],
                "memory_usage_mb": row["memory_usage_mb"],
                "total_steps": row["total_steps"],
                "error_message": row["error_message"],
            }

    async def list_run_steps(self, run_id: str) -> List[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows: Iterable[Record] = await conn.fetch(
                """
                SELECT step_name, step_index, status, output, raw_response, cost_usd,
                       token_counts, execution_time_ms, created_at
                FROM steps WHERE run_id = $1 ORDER BY step_index
                """,
                run_id,
            )
            results: List[JSONObject] = []
            for row in rows:
                results.append(
                    {
                        "step_name": row["step_name"],
                        "step_index": row["step_index"],
                        "status": row["status"],
                        "output": safe_deserialize(row["output"]),
                        "raw_response": safe_deserialize(row["raw_response"]),
                        "cost_usd": row["cost_usd"],
                        "token_counts": row["token_counts"],
                        "execution_time_ms": row["execution_time_ms"],
                        "created_at": row["created_at"],
                    }
                )
            return results

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False


__all__ = ["PostgresBackend"]
