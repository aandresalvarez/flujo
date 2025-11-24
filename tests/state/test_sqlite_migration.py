"""Schema migration tests for SQLiteBackend."""

from __future__ import annotations

from pathlib import Path
from typing import Set

import aiosqlite
import pytest

from flujo.state.backends.sqlite import SQLiteBackend

WORKFLOW_REQUIRED_COLUMNS: Set[str] = {
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
}

RUNS_REQUIRED_COLUMNS: Set[str] = {
    "run_id",
    "pipeline_id",
    "pipeline_name",
    "pipeline_version",
    "status",
    "created_at",
    "updated_at",
    "end_time",
    "total_cost",
    "final_context_blob",
    "execution_time_ms",
    "memory_usage_mb",
    "total_steps",
    "error_message",
}

WORKFLOW_INDEXES: Set[str] = {
    "idx_workflow_state_status",
    "idx_workflow_state_pipeline_id",
    "idx_workflow_state_created_at",
}

RUNS_INDEXES: Set[str] = {
    "idx_runs_status",
    "idx_runs_pipeline_id",
    "idx_runs_created_at",
    "idx_runs_pipeline_name",
}

pytestmark = pytest.mark.asyncio


async def _seed_legacy_schema(db_path: Path) -> None:
    """Create a legacy schema missing new columns to exercise migration."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE workflow_state (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                current_step_index INTEGER,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        await db.execute(
            """
            INSERT INTO workflow_state (
                run_id, pipeline_id, pipeline_name, pipeline_version,
                current_step_index, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_run",
                "legacy_pipeline",
                "LegacyPipeline",
                "0.1",
                0,
                "running",
                "",
                "",
            ),
        )

        await db.execute(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        await db.execute(
            """
            INSERT INTO runs (
                run_id, pipeline_id, pipeline_name, pipeline_version,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_run",
                "legacy_pipeline",
                "LegacyPipeline",
                "0.1",
                "running",
                "",
                "",
            ),
        )
        await db.commit()


async def _table_columns(db: aiosqlite.Connection, table: str) -> Set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    await cursor.close()
    return {row[1] for row in rows}


async def _indexes(db: aiosqlite.Connection, table: str) -> Set[str]:
    cursor = await db.execute(f"PRAGMA index_list('{table}')")
    rows = await cursor.fetchall()
    await cursor.close()
    return {row[1] for row in rows}


async def test_sqlite_migration_adds_columns_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_state.db"
    await _seed_legacy_schema(db_path)

    backend = SQLiteBackend(db_path)
    try:
        # Trigger initialization + migration path
        states = await backend.list_states()
    finally:
        await backend.close()

    async with aiosqlite.connect(db_path) as db:
        workflow_columns = await _table_columns(db, "workflow_state")
        runs_columns = await _table_columns(db, "runs")
        workflow_indexes = await _indexes(db, "workflow_state")
        runs_indexes = await _indexes(db, "runs")

    assert WORKFLOW_REQUIRED_COLUMNS.issubset(workflow_columns)
    assert RUNS_REQUIRED_COLUMNS.issubset(runs_columns)
    assert WORKFLOW_INDEXES.issubset(workflow_indexes)
    assert RUNS_INDEXES.issubset(runs_indexes)
    assert any(state["run_id"] == "legacy_run" for state in states)
