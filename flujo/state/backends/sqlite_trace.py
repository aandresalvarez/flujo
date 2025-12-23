"""SQLite trace mixin for span persistence and retrieval."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Callable, Awaitable, Protocol

from .sqlite_core import _fast_json_dumps
from ...type_definitions.common import JSONObject

if TYPE_CHECKING:  # pragma: no cover
    from aiosqlite import Connection

    class _SQLiteBackendDeps(Protocol):
        _lock: Any

        async def _ensure_init(self) -> None: ...

        async def _create_connection(self) -> "Connection": ...

        async def _with_retries(
            self, coro_func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
        ) -> Any: ...

        def _extract_spans_from_tree(
            self,
            trace: JSONObject,
            run_id: str,
            max_depth: int = 100,
        ) -> List[Tuple[str, str, Optional[str], str, float, Optional[float], str, str]]: ...

        def _reconstruct_trace_tree(
            self,
            spans_data: List[Tuple[str, Optional[str], str, float, Optional[float], str, str]],
        ) -> Optional[JSONObject]: ...


class SQLiteTraceMixin:
    async def save_spans(self: "_SQLiteBackendDeps", run_id: str, spans: list[JSONObject]) -> None:
        """Persist normalized spans for a given run_id."""
        if not run_id or not spans:
            return
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                conn = await self._create_connection()
                try:
                    db = conn
                    await db.execute("PRAGMA foreign_keys = ON")

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

                    try:
                        now = datetime.now(timezone.utc).isoformat()
                        await db.execute(
                            """
                            INSERT OR IGNORE INTO runs (
                                run_id, pipeline_id, pipeline_name, pipeline_version, status,
                                created_at, updated_at
                            ) VALUES (?, 'unknown', ?, ?, 'running', ?, ?)
                            """,
                            (
                                run_id,
                                pipeline_name or "unknown",
                                pipeline_version or "latest",
                                now,
                                now,
                            ),
                        )
                    except sqlite3.Error:
                        pass

                    spans_to_insert: List[
                        Tuple[str, str, Optional[str], str, float, Optional[float], str, str]
                    ] = []
                    for span in spans:
                        try:
                            span_id = str(span.get("span_id", ""))
                            if not span_id:
                                continue
                            parent_span_id_val = span.get("parent_span_id")
                            parent_span_id = (
                                str(parent_span_id_val) if parent_span_id_val is not None else None
                            )
                            name = str(span.get("name", "")) or "span"
                            start_time = float(span.get("start_time", 0.0))
                            end_raw = span.get("end_time")
                            end_time = float(end_raw) if end_raw is not None else None
                            status = str(span.get("status", "running"))
                            attrs = span.get("attributes", {})
                            attrs_json = _fast_json_dumps(attrs if isinstance(attrs, dict) else {})
                        except (ValueError, TypeError):
                            continue

                        spans_to_insert.append(
                            (
                                span_id,
                                run_id,
                                parent_span_id,
                                name,
                                start_time,
                                end_time,
                                status,
                                attrs_json,
                            )
                        )

                    if spans_to_insert:
                        await db.executemany(
                            """
                            INSERT INTO spans (
                                span_id, run_id, parent_span_id, name, start_time,
                                end_time, status, attributes, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                            """,
                            spans_to_insert,
                        )
                        await db.commit()
                        try:
                            from flujo.infra.audit import log_audit as _audit

                            _audit("spans_saved", run_id=run_id, span_count=len(spans_to_insert))
                        except Exception:
                            pass
                finally:
                    await conn.close()

            await self._with_retries(_save)

    async def save_trace(self: "_SQLiteBackendDeps", run_id: str, trace: JSONObject) -> None:
        """Persist a trace tree as normalized spans for a given run_id."""
        await self._ensure_init()
        async with self._lock:

            async def _save() -> None:
                conn = await self._create_connection()
                try:
                    db = conn
                    await db.execute("PRAGMA foreign_keys = ON")

                    try:
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
                    except sqlite3.Error:
                        pass

                    spans_to_insert = self._extract_spans_from_tree(trace, run_id)

                    if spans_to_insert:
                        await db.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))

                        await db.executemany(
                            """
                            INSERT INTO spans (
                                span_id, run_id, parent_span_id, name, start_time,
                                end_time, status, attributes, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                            """,
                            spans_to_insert,
                        )
                        await db.commit()
                        try:
                            from flujo.infra.audit import log_audit as _audit

                            _audit("trace_saved", run_id=run_id, span_count=len(spans_to_insert))
                        except Exception:
                            pass
                finally:
                    await conn.close()

            await self._with_retries(_save)

    def _extract_spans_from_tree(
        self: "_SQLiteBackendDeps", trace: JSONObject, run_id: str, max_depth: int = 100
    ) -> List[Tuple[str, str, Optional[str], str, float, Optional[float], str, str]]:
        """Extract all spans from a trace tree for batch insertion."""
        spans: List[Tuple[str, str, Optional[str], str, float, Optional[float], str, str]] = []

        # Handle empty or invalid trace data
        if not trace or not isinstance(trace, dict):
            return spans

        def extract_span_recursive(
            span_data: JSONObject,
            parent_span_id: Optional[str] = None,
            depth: int = 0,
        ) -> None:
            # Check depth limit to prevent stack overflow
            if depth > max_depth:
                from flujo.infra import telemetry

                telemetry.logfire.warn(
                    f"Trace tree depth limit ({max_depth}) exceeded for run_id {run_id}"
                )
                return

            # Validate required fields
            if (
                not isinstance(span_data, dict)
                or "span_id" not in span_data
                or "name" not in span_data
            ):
                return

            try:
                start_time = float(span_data.get("start_time", 0.0))
            except (ValueError, TypeError):
                logging.warning(
                    f"Skipping span with invalid start_time for run_id={run_id}, span_id={span_data.get('span_id')}"
                )
                return
            try:
                end_time = (
                    float(span_data["end_time"]) if span_data.get("end_time") is not None else None
                )
            except (ValueError, TypeError):
                logging.warning(
                    f"Skipping span with invalid end_time for run_id={run_id}, span_id={span_data.get('span_id')}"
                )
                return

            span_tuple: Tuple[str, str, Optional[str], str, float, Optional[float], str, str] = (
                str(span_data.get("span_id", "")),
                run_id,
                parent_span_id,
                str(span_data.get("name", "")),
                start_time,
                end_time,
                str(span_data.get("status", "running")),
                _fast_json_dumps(span_data.get("attributes", {})),
            )
            spans.append(span_tuple)

            # Process children recursively
            for child in span_data.get("children", []):
                extract_span_recursive(child, span_data.get("span_id"), depth + 1)

        extract_span_recursive(trace)
        return spans

    def _reconstruct_trace_tree(
        self: "_SQLiteBackendDeps",
        spans_data: List[Tuple[str, Optional[str], str, float, Optional[float], str, str]],
    ) -> Optional[JSONObject]:
        """Reconstruct a hierarchical trace tree from flat spans data."""
        spans_map: Dict[str, JSONObject] = {}
        root_spans: List[JSONObject] = []

        # First pass: create a map of all spans by ID
        for row in spans_data:
            (
                span_id,
                parent_span_id,
                name,
                start_time,
                end_time,
                status,
                attributes,
            ) = row
            span_data: JSONObject = {
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "attributes": json.loads(attributes) if attributes else {},
                "children": [],
            }
            spans_map[span_id] = span_data

        # Second pass: build the tree hierarchy
        for span_id, span_data in spans_map.items():
            parent_id = span_data.get("parent_span_id")
            if parent_id and parent_id in spans_map:
                spans_map[parent_id]["children"].append(span_data)
            else:
                root_spans.append(span_data)

        if len(root_spans) > 1:
            from flujo.infra import telemetry

            telemetry.logfire.warn(
                f"Trace for run_id has multiple root spans ({len(root_spans)}). Using the first one."
            )

        return root_spans[0] if root_spans else None

    async def get_trace(self: "_SQLiteBackendDeps", run_id: str) -> Optional[JSONObject]:
        """Retrieve and reconstruct the trace tree for a given run_id. Audit log access."""
        await self._ensure_init()
        async with self._lock:
            # Structured audit log for trace access
            try:
                from flujo.infra.audit import log_audit as _audit

                _audit("trace_access", run_id=run_id)
            except Exception:
                pass
            conn = await self._create_connection()
            try:
                db = conn
                await db.execute("PRAGMA foreign_keys = ON")
                async with db.execute(
                    """
                    SELECT span_id, parent_span_id, name, start_time, end_time,
                           status, attributes
                    FROM spans WHERE run_id = ? ORDER BY start_time
                    """,
                    (run_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
                    rows_typed: List[
                        Tuple[str, Optional[str], str, float, Optional[float], str, str]
                    ] = [
                        (
                            str(r[0]),
                            str(r[1]) if r[1] is not None else None,
                            str(r[2]),
                            float(r[3]),
                            float(r[4]) if r[4] is not None else None,
                            str(r[5]),
                            str(r[6]),
                        )
                        for r in rows
                    ]
                    if not rows_typed:
                        return None
                    return self._reconstruct_trace_tree(rows_typed)
            finally:
                await conn.close()

    async def get_spans(
        self: "_SQLiteBackendDeps",
        run_id: str,
        status: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[JSONObject]:
        """Get individual spans with optional filtering. Audit log export."""
        await self._ensure_init()
        async with self._lock:
            try:
                from flujo.infra.audit import log_audit as _audit

                _audit("spans_exported", run_id=run_id, status=status, name=name)
            except Exception:
                pass
            conn = await self._create_connection()
            try:
                db = conn
                await db.execute("PRAGMA foreign_keys = ON")
                query = """
                    SELECT span_id, parent_span_id, name, start_time, end_time,
                           status, attributes
                    FROM spans WHERE run_id = ?
                """
                params: List[Any] = [run_id]
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if name:
                    query += " AND name = ?"
                    params.append(name)
                query += " ORDER BY start_time"
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    results: List[JSONObject] = []
                    for r in rows:
                        (
                            span_id,
                            parent_span_id,
                            name,
                            start_time,
                            end_time,
                            status,
                            attributes,
                        ) = r
                        results.append(
                            {
                                "span_id": str(span_id),
                                "parent_span_id": str(parent_span_id)
                                if parent_span_id is not None
                                else None,
                                "name": str(name),
                                "start_time": float(start_time),
                                "end_time": float(end_time) if end_time is not None else None,
                                "status": str(status),
                                "attributes": json.loads(attributes) if attributes else {},
                            }
                        )
                    return results
            finally:
                await conn.close()

    async def get_span_statistics(
        self: "_SQLiteBackendDeps",
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> JSONObject:
        """Get aggregated span statistics."""
        await self._ensure_init()
        async with self._lock:
            conn = await self._create_connection()
            try:
                db = conn
                await db.execute("PRAGMA foreign_keys = ON")
                query = """
                    SELECT s.name, s.status, s.start_time, s.end_time,
                           r.pipeline_name
                    FROM spans s
                    JOIN runs r ON s.run_id = r.run_id
                    WHERE s.end_time IS NOT NULL
                """
                params: List[Any] = []
                if pipeline_name:
                    query += " AND r.pipeline_name = ?"
                    params.append(pipeline_name)
                if time_range:
                    start_time, end_time = time_range
                    query += " AND s.start_time >= ? AND s.start_time <= ?"
                    params.extend([start_time, end_time])
                async with db.execute(query, params) as cursor:
                    rows = list(await cursor.fetchall())
                    stats: JSONObject = {
                        "total_spans": len(rows),
                        "by_name": {},
                        "by_status": {},
                        "avg_duration_by_name": {},
                    }
                    for r in rows:
                        name, status, start_time, end_time, pipeline_name = r
                        duration = (
                            float(end_time) - float(start_time) if end_time is not None else 0.0
                        )
                        # Count by name
                        if name not in stats["by_name"]:
                            stats["by_name"][name] = 0
                        stats["by_name"][name] += 1
                        # Count by status
                        if status not in stats["by_status"]:
                            stats["by_status"][status] = 0
                        stats["by_status"][status] += 1
                        # Average duration by name
                        if name not in stats["avg_duration_by_name"]:
                            stats["avg_duration_by_name"][name] = {
                                "total": 0.0,
                                "count": 0,
                            }
                        stats["avg_duration_by_name"][name]["total"] += duration
                        stats["avg_duration_by_name"][name]["count"] += 1
                    for name, data in stats["avg_duration_by_name"].items():
                        if data["count"] > 0:
                            data["average"] = data["total"] / data["count"]
                        else:
                            data["average"] = 0.0
                    return stats
            finally:
                await conn.close()
