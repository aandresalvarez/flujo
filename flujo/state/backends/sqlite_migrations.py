from __future__ import annotations

import sqlite3
from typing import Iterable

import aiosqlite


LATEST_SCHEMA_VERSION = 1


async def _get_user_version(db: aiosqlite.Connection) -> int:
    cur = await db.execute("PRAGMA user_version")
    row = await cur.fetchone()
    await cur.close()
    try:
        return int(row[0]) if row is not None else 0
    except Exception:
        return 0


async def _set_user_version(db: aiosqlite.Connection, version: int) -> None:
    await db.execute(f"PRAGMA user_version = {int(version)}")


async def _get_table_columns(db: aiosqlite.Connection, table: str) -> set[str]:
    try:
        cur = await db.execute(f"PRAGMA table_info({table})")
        rows = await cur.fetchall()
        await cur.close()
        return {str(r[1]) for r in rows}
    except sqlite3.OperationalError:
        return set()


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


async def _add_columns_if_missing(
    db: aiosqlite.Connection, *, table: str, columns: Iterable[tuple[str, str]]
) -> None:
    existing = await _get_table_columns(db, table)
    if not existing:
        return
    for column_name, column_def in columns:
        if column_name in existing:
            continue
        await db.execute(
            f"ALTER TABLE {table} ADD COLUMN {_quote_identifier(column_name)} {column_def}"
        )


async def _migrate_0_to_1(db: aiosqlite.Connection) -> None:
    # workflow_state columns
    await _add_columns_if_missing(
        db,
        table="workflow_state",
        columns=[
            ("pipeline_name", "TEXT NOT NULL DEFAULT ''"),
            ("pipeline_version", "TEXT NOT NULL DEFAULT '1.0'"),
            ("current_step_index", "INTEGER NOT NULL DEFAULT 0"),
            ("pipeline_context", "TEXT"),
            ("last_step_output", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'running'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ("total_steps", "INTEGER DEFAULT 0"),
            ("error_message", "TEXT"),
            ("execution_time_ms", "INTEGER"),
            ("memory_usage_mb", "REAL"),
            ("step_history", "TEXT"),
            ("metadata", "TEXT"),
            ("is_background_task", "INTEGER DEFAULT 0"),
            ("parent_run_id", "TEXT"),
            ("task_id", "TEXT"),
            ("background_error", "TEXT"),
        ],
    )

    # Backfill required defaults when present
    try:
        cols = await _get_table_columns(db, "workflow_state")
        if cols:
            if "current_step_index" in cols:
                await db.execute(
                    "UPDATE workflow_state SET current_step_index = 0 WHERE current_step_index IS NULL"
                )
            if "pipeline_context" in cols:
                await db.execute(
                    "UPDATE workflow_state SET pipeline_context = '{}' WHERE pipeline_context IS NULL"
                )
            if "status" in cols:
                await db.execute(
                    "UPDATE workflow_state SET status = 'running' WHERE status IS NULL"
                )
            if "pipeline_name" in cols:
                await db.execute(
                    "UPDATE workflow_state SET pipeline_name = pipeline_id WHERE pipeline_name = ''"
                )
            if "pipeline_version" in cols:
                await db.execute(
                    "UPDATE workflow_state SET pipeline_version = '1.0' WHERE pipeline_version = ''"
                )
            if "created_at" in cols:
                await db.execute(
                    "UPDATE workflow_state SET created_at = datetime('now') WHERE created_at = ''"
                )
            if "updated_at" in cols:
                await db.execute(
                    "UPDATE workflow_state SET updated_at = datetime('now') WHERE updated_at = ''"
                )
    except Exception:
        pass

    # steps.raw_response (FSD-013)
    try:
        cols = await _get_table_columns(db, "steps")
        if cols and "raw_response" not in cols:
            await db.execute("ALTER TABLE steps ADD COLUMN raw_response TEXT")
    except sqlite3.OperationalError:
        pass

    # runs columns (older DBs)
    await _add_columns_if_missing(
        db,
        table="runs",
        columns=[
            ("pipeline_id", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("pipeline_name", "TEXT NOT NULL DEFAULT ''"),
            ("pipeline_version", "TEXT NOT NULL DEFAULT '1.0'"),
            ("status", "TEXT NOT NULL DEFAULT 'running'"),
            ("created_at", "TEXT NOT NULL DEFAULT ''"),
            ("updated_at", "TEXT NOT NULL DEFAULT ''"),
            ("end_time", "TEXT"),
            ("total_cost", "REAL"),
            ("final_context_blob", "TEXT"),
            ("execution_time_ms", "INTEGER"),
            ("memory_usage_mb", "REAL"),
            ("total_steps", "INTEGER DEFAULT 0"),
            ("error_message", "TEXT"),
        ],
    )

    # Backfill run defaults when present
    try:
        cols = await _get_table_columns(db, "runs")
        if cols:
            if "pipeline_id" in cols:
                await db.execute(
                    "UPDATE runs SET pipeline_id = 'unknown' WHERE pipeline_id IS NULL OR pipeline_id = ''"
                )
            if "pipeline_name" in cols:
                await db.execute(
                    "UPDATE runs SET pipeline_name = pipeline_id "
                    "WHERE (pipeline_name IS NULL OR pipeline_name = '') AND pipeline_id IS NOT NULL"
                )
            if "pipeline_version" in cols:
                await db.execute(
                    "UPDATE runs SET pipeline_version = '1.0' "
                    "WHERE pipeline_version IS NULL OR pipeline_version = ''"
                )
            if "status" in cols:
                await db.execute(
                    "UPDATE runs SET status = 'running' WHERE status IS NULL OR status = ''"
                )
            if "created_at" in cols:
                await db.execute("UPDATE runs SET created_at = '' WHERE created_at IS NULL")
            if "updated_at" in cols:
                await db.execute("UPDATE runs SET updated_at = '' WHERE updated_at IS NULL")
    except Exception:
        pass


async def _schema_needs_repair_v1(db: aiosqlite.Connection) -> bool:
    """Return True when a v1 database is missing required columns.

    Some tests (and real-world corruption cases) can drop individual columns while
    leaving `PRAGMA user_version` untouched. Since `CREATE TABLE IF NOT EXISTS`
    does not retrofit missing columns, we need a lightweight "repair" check even
    when the schema version is already current.
    """

    required_by_table: dict[str, set[str]] = {
        "runs": {
            "run_id",
            "pipeline_id",
            "pipeline_name",
            "pipeline_version",
            "status",
            "created_at",
            "updated_at",
            "execution_time_ms",
            "memory_usage_mb",
            "total_steps",
            "error_message",
        },
        "workflow_state": {
            "run_id",
            "pipeline_id",
            "pipeline_name",
            "pipeline_version",
            "current_step_index",
            "pipeline_context",
            "last_step_output",
            "step_history",
            "status",
            "created_at",
            "updated_at",
            "total_steps",
            "error_message",
            "execution_time_ms",
            "memory_usage_mb",
            "metadata",
            "is_background_task",
            "parent_run_id",
            "task_id",
            "background_error",
        },
        "steps": {
            "id",
            "run_id",
            "step_name",
            "step_index",
            "status",
            "output",
            "raw_response",
            "cost_usd",
            "token_counts",
            "execution_time_ms",
            "created_at",
        },
    }

    for table, required in required_by_table.items():
        cols = await _get_table_columns(db, table)
        if cols and not required.issubset(cols):
            return True
    return False


async def apply_sqlite_migrations(db: aiosqlite.Connection) -> None:
    """Apply versioned SQLite schema migrations using PRAGMA user_version."""
    current = await _get_user_version(db)
    if current < 1:
        await _migrate_0_to_1(db)
        await _set_user_version(db, 1)
        current = 1
    elif current == 1 and await _schema_needs_repair_v1(db):
        # Self-heal missing columns even when user_version is already current.
        await _migrate_0_to_1(db)
    if current != LATEST_SCHEMA_VERSION:
        # Forward compatibility: do not silently continue on unknown future schema.
        raise RuntimeError(
            f"SQLite schema version {current} is newer than supported {LATEST_SCHEMA_VERSION}"
        )
