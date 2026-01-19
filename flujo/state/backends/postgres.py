from __future__ import annotations

import asyncio
import json
import importlib
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

from flujo.state.backends.base import StateBackend, _to_jsonable
from flujo.type_definitions.common import JSONObject
from flujo.utils.serialization import safe_deserialize

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg
    from asyncpg import Pool, Record
else:  # pragma: no cover - runtime checked import
    asyncpg = None
    Pool = Any
    Record = Any

# Advisory lock key used to serialize schema migrations across concurrent workers/pods.
# Must fit in signed bigint; keep stable for all deployments.
MIGRATION_LOCK_KEY = 0xF1F0C0DE


def _load_asyncpg() -> Any:
    spec = importlib.util.find_spec("asyncpg")
    if spec is None:
        raise RuntimeError("asyncpg is required. Install with `pip install flujo[postgres]`.")
    module = importlib.import_module("asyncpg")
    return module


def _jsonb(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(_to_jsonable(value))


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse ISO string or datetime to datetime object for asyncpg.

    Asyncpg requires actual datetime objects for TIMESTAMPTZ columns,
    but StateManager often passes ISO strings from serialization.

    Args:
        value: ISO string, datetime object, or None

    Returns:
        datetime object or None.

    Raises:
        ValueError: If a string cannot be parsed as ISO timestamp.
        TypeError: If the value type is unsupported.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid timestamp string: {value!r}.") from exc
    raise TypeError(f"Unsupported timestamp type: {type(value).__name__}.")


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
        self._init_lock: Optional[asyncio.Lock] = None
        self._initialized = False
        self._max_span_depth = 100

    def _get_init_lock(self) -> asyncio.Lock:
        """Lazily create the init lock on first access."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    async def _ensure_pool(self) -> Pool:
        if self._pool is not None:
            return self._pool
        async with self._get_init_lock():
            if self._pool is None:
                pg = _load_asyncpg()
                pool = await pg.create_pool(
                    self._dsn,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                )
                self._pool = pool
        assert self._pool is not None
        return self._pool

    async def _ensure_init(self) -> None:
        pool = await self._ensure_pool()
        if self._initialized:
            return
        async with self._get_init_lock():
            if self._initialized:
                return
            if self._auto_migrate:
                await self._init_schema(pool)
            else:
                await self._verify_schema(pool)
            self._initialized = True

    async def _verify_schema(self, pool: Pool) -> None:
        async with pool.acquire() as conn:
            required_tables = [
                "workflow_state",
                "runs",
                "steps",
                "traces",
                "spans",
                "flujo_schema_versions",
            ]
            missing: list[str] = []
            for table in required_tables:
                exists = await conn.fetchval("SELECT to_regclass($1)", table)
                if exists is None:
                    missing.append(table)
            if missing:
                raise RuntimeError(
                    f"Missing required Postgres tables: {', '.join(missing)}. "
                    "Run `flujo migrate` or enable FLUJO_AUTO_MIGRATE."
                )

    async def _init_schema(self, pool: Pool) -> None:
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Serialize migrations across processes/containers using a Postgres advisory lock.
                # This prevents migration stampedes on scale-up.
                try:
                    await conn.execute("SELECT pg_advisory_xact_lock($1)", MIGRATION_LOCK_KEY)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Advisory lock acquisition failed, continuing without serialization: %s",
                        e,
                    )
                # Create base schema (version 1)
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
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
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
            await conn.execute(
                """
                INSERT INTO flujo_schema_versions (version, applied_at)
                VALUES (1, NOW())
                ON CONFLICT (version) DO NOTHING
                """
            )
            # Apply pending migrations after base schema creation
            await self._apply_pending_migrations(conn)

    async def _apply_pending_migrations(self, conn: Any) -> None:
        """Apply pending migration files from flujo/state/migrations/."""
        # Find migration files
        # postgres.py is in flujo/state/backends/, so go up to flujo/state/ then into migrations/
        migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
        if not migrations_dir.exists():
            return

        migration_files: list[Tuple[int, Path]] = []
        for path in migrations_dir.glob("*.sql"):
            stem = path.stem.split("_", 1)[0]
            if stem.isdigit():
                version = int(stem)
                # Skip version 1 as it's already applied by _init_schema
                if version > 1:
                    migration_files.append((version, path))
        migration_files.sort(key=lambda item: item[0])

        if not migration_files:
            return

        # Get already applied versions
        applied_rows = await conn.fetch("SELECT version FROM flujo_schema_versions")
        applied_versions = {int(row["version"]) for row in applied_rows}

        # Apply pending migrations
        for version, path in migration_files:
            if version in applied_versions:
                continue
            sql = path.read_text()
            # Execute each migration inside its own transaction to avoid partial application.
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO flujo_schema_versions (version) VALUES ($1) ON CONFLICT DO NOTHING",
                    version,
                )

    async def persist_evaluation(
        self,
        run_id: str,
        score: float,
        feedback: str | None = None,
        step_name: str | None = None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Persist a shadow evaluation result into Postgres."""
        await self._ensure_init()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evaluations (run_id, step_name, score, feedback, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                run_id,
                step_name,
                score,
                feedback,
                _jsonb(metadata),
            )

    async def list_evaluations(
        self,
        limit: int = 20,
        run_id: str | None = None,
    ) -> list[JSONObject]:
        """Return recent evaluations."""
        await self._ensure_init()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if run_id:
                rows = await conn.fetch(
                    """
                    SELECT run_id, step_name, score, feedback, metadata, created_at
                    FROM evaluations
                    WHERE run_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    run_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT run_id, step_name, score, feedback, metadata, created_at
                    FROM evaluations
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
        records: list[JSONObject] = []
        for row in rows:
            records.append(
                {
                    "run_id": row["run_id"],
                    "step_name": row["step_name"],
                    "score": row["score"],
                    "feedback": row["feedback"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"],
                }
            )
        return records

    def _extract_spans_from_tree(
        self, trace: JSONObject, run_id: str, max_depth: int = 100
    ) -> List[
        Tuple[
            str,
            str,
            Optional[str],
            str,
            float,
            Optional[float],
            str,
            datetime,
            JSONObject,
        ]
    ]:
        """Flatten a trace tree into span tuples for insertion."""
        spans: List[
            Tuple[
                str,
                str,
                Optional[str],
                str,
                float,
                Optional[float],
                str,
                datetime,
                JSONObject,
            ]
        ] = []

        if not trace or not isinstance(trace, dict):
            return spans

        def extract_span_recursive(
            span_data: JSONObject, parent_span_id: Optional[str], depth: int
        ) -> None:
            if depth > max_depth:
                return
            if (
                not isinstance(span_data, dict)
                or "span_id" not in span_data
                or "name" not in span_data
            ):
                return
            try:
                start_time = float(span_data.get("start_time", 0.0))
            except (ValueError, TypeError):
                return
            try:
                end_time = (
                    float(span_data["end_time"]) if span_data.get("end_time") is not None else None
                )
            except (ValueError, TypeError):
                return

            created_at = datetime.now(timezone.utc)
            spans.append(
                (
                    run_id,
                    str(span_data.get("span_id", "")),
                    parent_span_id,
                    str(span_data.get("name", "")),
                    start_time,
                    end_time,
                    str(span_data.get("status", "running")),
                    created_at,
                    span_data.get("attributes", {}),
                )
            )
            for child in span_data.get("children", []):
                extract_span_recursive(child, str(span_data.get("span_id")), depth + 1)

        extract_span_recursive(trace, None, 0)
        return spans

    async def save_state(self, run_id: str, state: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            created_at = _parse_timestamp(state.get("created_at")) or datetime.now(timezone.utc)
            updated_at = _parse_timestamp(state.get("updated_at")) or datetime.now(timezone.utc)
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
                "is_background_task": (
                    bool(record["is_background_task"])
                    if record["is_background_task"] is not None
                    else False
                ),
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
            raw = safe_deserialize(record["trace_data"])
            if isinstance(raw, dict):
                return raw
            return None

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            query = """
                SELECT span_id, parent_span_id, name, start_time, end_time,
                       status, attributes, created_at
                FROM spans WHERE run_id = $1
            """
            params: List[Any] = [run_id]
            param_idx = 2
            if status:
                query += f" AND status = ${param_idx}"
                params.append(status)
                param_idx += 1
            if name:
                query += f" AND name = ${param_idx}"
                params.append(name)
            query += " ORDER BY start_time"
            rows = await conn.fetch(query, *params)
            results: List[JSONObject] = []
            for row in rows:
                results.append(
                    {
                        "span_id": str(row["span_id"]),
                        "parent_span_id": (
                            str(row["parent_span_id"])
                            if row["parent_span_id"] is not None
                            else None
                        ),
                        "name": str(row["name"]),
                        "start_time": float(row["start_time"]),
                        "end_time": (
                            float(row["end_time"]) if row["end_time"] is not None else None
                        ),
                        "status": str(row["status"]),
                        "attributes": (
                            safe_deserialize(row["attributes"]) if row["attributes"] else {}
                        ),
                        "created_at": row["created_at"],
                    }
                )
            return results

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> JSONObject:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            query = """
                SELECT s.name, s.status, s.start_time, s.end_time, r.pipeline_name
                FROM spans s
                JOIN runs r ON s.run_id = r.run_id
                WHERE s.end_time IS NOT NULL
            """
            params: List[Any] = []
            param_idx = 1
            if pipeline_name:
                query += f" AND r.pipeline_name = ${param_idx}"
                params.append(pipeline_name)
                param_idx += 1
            if time_range:
                start_time, end_time = time_range
                query += f" AND s.start_time >= ${param_idx} AND s.start_time <= ${param_idx + 1}"
                params.extend([start_time, end_time])
            rows = await conn.fetch(query, *params)
            stats: JSONObject = {
                "total_spans": len(rows),
                "by_name": {},
                "by_status": {},
                "avg_duration_by_name": {},
            }
            for row in rows:
                name = str(row["name"])
                status = str(row["status"])
                start_time_val = float(row["start_time"])
                end_time_val = float(row["end_time"]) if row["end_time"] is not None else 0.0
                duration = end_time_val - start_time_val

                stats["by_name"][name] = stats["by_name"].get(name, 0) + 1
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                if name not in stats["avg_duration_by_name"]:
                    stats["avg_duration_by_name"][name] = {"total": 0.0, "count": 0}
                stats["avg_duration_by_name"][name]["total"] += duration
                stats["avg_duration_by_name"][name]["count"] += 1
            for name, data in stats["avg_duration_by_name"].items():
                count = data["count"]
                data["average"] = data["total"] / count if count > 0 else 0.0
            return stats

    async def save_trace(self, run_id: str, trace: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            now = datetime.now(timezone.utc)
            spans = self._extract_spans_from_tree(trace, run_id, self._max_span_depth)
            async with conn.transaction():
                # Ensure a runs record exists for FK integrity (unknowns are safe defaults)
                await conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, pipeline_id, pipeline_name, pipeline_version, status,
                        created_at, updated_at
                    ) VALUES ($1, 'unknown', 'unknown', 'latest', 'running', $2, $2)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    run_id,
                    now,
                )
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
                await conn.execute("DELETE FROM spans WHERE run_id = $1", run_id)
                if spans:
                    await conn.executemany(
                        """
                        INSERT INTO spans (
                            run_id, span_id, parent_span_id, name, start_time, end_time,
                            status, attributes, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                        """,
                        [
                            (
                                span_run_id,
                                span_id,
                                parent_span_id,
                                name,
                                start_time,
                                end_time,
                                status,
                                _jsonb(attributes),
                                created_at,
                            )
                            for (
                                span_run_id,
                                span_id,
                                parent_span_id,
                                name,
                                start_time,
                                end_time,
                                status,
                                created_at,
                                attributes,
                            ) in spans
                        ],
                    )

    async def save_spans(self, run_id: str, spans: list[JSONObject]) -> None:
        if not run_id or not spans:
            return
        await self._ensure_init()
        assert self._pool is not None
        pipeline_name: Optional[str] = None
        pipeline_version: Optional[str] = None
        for span in spans:
            attrs = span.get("attributes")
            if not isinstance(attrs, dict):
                continue
            if pipeline_name is None:
                val = attrs.get("flujo.pipeline.name")
                if isinstance(val, str) and val:
                    pipeline_name = val
            if pipeline_version is None:
                val = attrs.get("flujo.pipeline.version")
                if isinstance(val, str) and val:
                    pipeline_version = val
            if pipeline_name and pipeline_version:
                break

        rows: List[Tuple[str, Optional[str], str, float, Optional[float], str, Optional[str]]] = []
        for span in spans:
            try:
                span_id = str(span.get("span_id", ""))
                if not span_id:
                    continue
                parent_span_id_val = span.get("parent_span_id")
                parent_span_id = str(parent_span_id_val) if parent_span_id_val is not None else None
                name = str(span.get("name", "")) or "span"
                start_time = float(span.get("start_time", 0.0))
                end_raw = span.get("end_time")
                end_time = float(end_raw) if end_raw is not None else None
                status = str(span.get("status", "running"))
                attrs = span.get("attributes", {})
                attrs_json = _jsonb(attrs if isinstance(attrs, dict) else {})
            except (ValueError, TypeError):
                continue

            rows.append(
                (
                    span_id,
                    parent_span_id,
                    name,
                    start_time,
                    end_time,
                    status,
                    attrs_json,
                )
            )

        if not rows:
            return

        async with self._pool.acquire() as conn:
            now = datetime.now(timezone.utc)
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, pipeline_id, pipeline_name, pipeline_version, status,
                        created_at, updated_at
                    ) VALUES ($1, 'unknown', $2, $3, 'running', $4, $4)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    run_id,
                    pipeline_name or "unknown",
                    pipeline_version or "latest",
                    now,
                )
                await conn.executemany(
                    """
                    INSERT INTO spans (
                        run_id, span_id, parent_span_id, name, start_time, end_time,
                        status, attributes, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                    """,
                    [
                        (
                            run_id,
                            span_id,
                            parent_span_id,
                            name,
                            start_time,
                            end_time,
                            status,
                            attrs_json,
                            now,
                        )
                        for (
                            span_id,
                            parent_span_id,
                            name,
                            start_time,
                            end_time,
                            status,
                            attrs_json,
                        ) in rows
                    ],
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
                _parse_timestamp(run_data.get("created_at")) or datetime.now(timezone.utc),
                _parse_timestamp(run_data.get("updated_at")) or datetime.now(timezone.utc),
                run_data.get("execution_time_ms"),
                run_data.get("memory_usage_mb"),
                run_data.get("total_steps", 0),
                run_data.get("error_message"),
            )

    async def save_step_result(self, step_data: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
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
                    _parse_timestamp(step_data.get("created_at")) or datetime.now(timezone.utc),
                )
                await conn.execute(
                    "UPDATE runs SET updated_at = NOW() WHERE run_id = $1",
                    step_data["run_id"],
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
                _parse_timestamp(end_data.get("updated_at")) or datetime.now(timezone.utc),
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

    async def set_system_state(self, key: str, value: JSONObject) -> None:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                """,
                key,
                _jsonb(value),
            )

    async def get_system_state(self, key: str) -> Optional[JSONObject]:
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT key, value, updated_at FROM system_state WHERE key = $1",
                key,
            )
            if row is None:
                return None
            stored_value = row["value"]
            deserialized = safe_deserialize(stored_value) if stored_value is not None else None
            return {
                "key": row["key"],
                "value": deserialized,
                "updated_at": row["updated_at"],
            }

    async def list_workflows(
        self,
        status: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[JSONObject]:
        """Enhanced workflow listing with additional filters and metadata."""
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # Build query with optional filters
            query_parts = [
                """
                SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                       current_step_index, status, created_at, updated_at,
                       total_steps, error_message, execution_time_ms, memory_usage_mb,
                       metadata, is_background_task, parent_run_id, task_id, background_error
                FROM workflow_state
                WHERE 1=1
                """
            ]
            params: List[Any] = []
            param_num = 1

            if status:
                query_parts.append(f" AND status = ${param_num}")
                params.append(status)
                param_num += 1

            if pipeline_id:
                query_parts.append(f" AND pipeline_id = ${param_num}")
                params.append(pipeline_id)
                param_num += 1

            query_parts.append(" ORDER BY created_at DESC")

            if limit is not None:
                query_parts.append(f" LIMIT ${param_num}")
                params.append(limit)
                param_num += 1
            if offset:
                query_parts.append(f" OFFSET ${param_num}")
                params.append(offset)

            query = "".join(query_parts)
            rows = await conn.fetch(query, *params)

            result: List[JSONObject] = []
            for row in rows:
                if row is None:
                    continue
                metadata = safe_deserialize(row["metadata"]) if row["metadata"] else {}
                result.append(
                    {
                        "run_id": row["run_id"],
                        "pipeline_id": row["pipeline_id"],
                        "pipeline_name": row["pipeline_name"],
                        "pipeline_version": row["pipeline_version"],
                        "current_step_index": row["current_step_index"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "total_steps": row["total_steps"] or 0,
                        "error_message": row["error_message"],
                        "execution_time_ms": row["execution_time_ms"],
                        "memory_usage_mb": row["memory_usage_mb"],
                        "metadata": metadata,
                        "is_background_task": (
                            bool(row["is_background_task"])
                            if row["is_background_task"] is not None
                            else False
                        ),
                        "parent_run_id": row["parent_run_id"],
                        "task_id": row["task_id"],
                        "background_error": row["background_error"],
                    }
                )
            return result

    async def list_runs(
        self,
        status: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        metadata_filter: Optional[JSONObject] = None,
    ) -> List[JSONObject]:
        """List runs from the structured schema with optional metadata filtering."""
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            query_parts = [
                """
                SELECT
                    r.run_id,
                    COALESCE(ws.pipeline_name, r.pipeline_name) AS pipeline_name,
                    COALESCE(ws.pipeline_version, r.pipeline_version) AS pipeline_version,
                    COALESCE(ws.status, r.status) AS status,
                    r.created_at,
                    r.updated_at,
                    r.execution_time_ms,
                    ws.metadata
                FROM runs r
                LEFT JOIN workflow_state ws ON ws.run_id = r.run_id
                WHERE 1=1
                """
            ]
            params: List[Any] = []
            param_num = 1

            if status:
                query_parts.append(f" AND COALESCE(ws.status, r.status) = ${param_num}")
                params.append(status)
                param_num += 1

            if pipeline_name:
                query_parts.append(
                    f" AND COALESCE(ws.pipeline_name, r.pipeline_name) = ${param_num}"
                )
                params.append(pipeline_name)
                param_num += 1

            if metadata_filter:
                query_parts.append(f" AND ws.metadata @> ${param_num}::jsonb")
                params.append(_jsonb(metadata_filter))
                param_num += 1

            query_parts.append(" ORDER BY r.created_at DESC")

            if limit is not None:
                query_parts.append(f" LIMIT ${param_num}")
                params.append(limit)
                param_num += 1
            if offset:
                query_parts.append(f" OFFSET ${param_num}")
                params.append(offset)

            query = "".join(query_parts)
            rows = await conn.fetch(query, *params)

            result: List[JSONObject] = []
            for row in rows:
                if row is None:
                    continue
                metadata = safe_deserialize(row["metadata"]) if row["metadata"] else {}
                result.append(
                    {
                        "run_id": row["run_id"],
                        "pipeline_name": row["pipeline_name"],
                        "pipeline_version": row["pipeline_version"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "metadata": metadata,
                        "start_time": row["created_at"],
                        "end_time": row["updated_at"],
                        "total_cost": (
                            row.get("total_cost")
                            if "total_cost" in row
                            else (row.get("cost_usd") if "cost_usd" in row else 0.0)
                        ),
                    }
                )
            return result

    async def get_workflow_stats(self) -> JSONObject:
        """Get comprehensive workflow statistics."""
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # Get total count
            total_workflows = await conn.fetchval("SELECT COUNT(*) FROM workflow_state") or 0

            # Get status counts
            status_rows = await conn.fetch(
                "SELECT status, COUNT(*) as count FROM workflow_state GROUP BY status"
            )
            status_counts: dict[str, int] = {
                row["status"]: row["count"] for row in status_rows if row is not None
            }

            # Get recent workflows (last 24 hours)
            recent_workflows_24h = (
                await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """
                )
                or 0
            )

            # Get average execution time
            avg_exec_time = (
                await conn.fetchval(
                    """
                    SELECT AVG(execution_time_ms)
                    FROM workflow_state
                    WHERE execution_time_ms IS NOT NULL
                    """
                )
                or 0
            )

            # Background task breakdown
            bg_rows = await conn.fetch(
                """
                SELECT status, COUNT(*) as count FROM workflow_state
                WHERE is_background_task = TRUE
                GROUP BY status
                """
            )
            bg_status_counts: dict[str, int] = {
                row["status"]: row["count"] for row in bg_rows if row is not None
            }

            return {
                "total_workflows": total_workflows,
                "status_counts": status_counts,
                "recent_workflows_24h": recent_workflows_24h,
                "average_execution_time_ms": avg_exec_time or 0,
                "background_status_counts": bg_status_counts,
            }

    async def get_failed_workflows(self, hours_back: int = 24) -> List[JSONObject]:
        """Get failed workflows from the last N hours."""
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT run_id, pipeline_id, pipeline_name, pipeline_version,
                       current_step_index, status, created_at, updated_at,
                       total_steps, error_message, execution_time_ms, memory_usage_mb
                FROM workflow_state
                WHERE status = $1
                AND updated_at >= NOW() - INTERVAL '1 hour' * $2
                ORDER BY updated_at DESC
                """,
                "failed",
                hours_back,
            )

            result: List[JSONObject] = []
            for row in rows:
                if row is None:
                    continue
                result.append(
                    {
                        "run_id": row["run_id"],
                        "pipeline_id": row["pipeline_id"],
                        "pipeline_name": row["pipeline_name"],
                        "pipeline_version": row["pipeline_version"],
                        "current_step_index": row["current_step_index"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "total_steps": row["total_steps"] or 0,
                        "error_message": row["error_message"],
                        "execution_time_ms": row["execution_time_ms"],
                        "memory_usage_mb": row["memory_usage_mb"],
                    }
                )
            return result

    async def cleanup_old_workflows(self, days_old: float = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        await self._ensure_init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # Count workflows to be deleted
            count = (
                await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM workflow_state
                    WHERE created_at < NOW() - INTERVAL '1 day' * $1
                    """,
                    days_old,
                )
                or 0
            )

            # Delete old workflows
            await conn.execute(
                """
                DELETE FROM workflow_state
                WHERE created_at < NOW() - INTERVAL '1 day' * $1
                """,
                days_old,
            )

            return int(count)

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False


__all__ = ["PostgresBackend"]
