from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from flujo.type_definitions.common import JSONObject

from .base import StateBackend, _serialize_for_json
from ._filters import metadata_contains


class InMemoryBackend(StateBackend):
    """Simple in-memory backend for testing and defaults.

    This backend mirrors the serialization logic of the persistent backends by
    storing a serialized copy of the workflow state. Values are serialized with
    ``_serialize_for_json`` on save and reconstructed with ``safe_deserialize`` when
    loaded.
    """

    def __init__(self) -> None:
        # Store serialized copies to mimic persistent backends
        self._store: dict[str, JSONObject] = {}
        self._system_state: dict[str, JSONObject] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create the lock on first access."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def save_state(self, run_id: str, state: JSONObject) -> None:
        async with self._get_lock():
            # Serialize state so custom types are handled consistently
            normalized = _serialize_for_json(state)
            serialized = json.loads(json.dumps(normalized, ensure_ascii=False))
            if not isinstance(serialized, dict):
                raise TypeError("Serialized state must be a mapping")
            self._store[run_id] = serialized

    async def load_state(self, run_id: str) -> Optional[JSONObject]:
        async with self._get_lock():
            stored = self._store.get(run_id)
            if stored is None:
                return None
            # Return a deserialized copy to avoid accidental mutation
            return deepcopy(stored)

    async def delete_state(self, run_id: str) -> None:
        async with self._get_lock():
            self._store.pop(run_id, None)

    async def get_trace(self, run_id: str) -> Any:
        """Retrieve trace data for a given run_id."""
        # InMemoryBackend doesn't support separate trace storage
        return None

    async def save_trace(self, run_id: str, trace: JSONObject) -> None:
        """Save trace data for a given run_id."""
        # InMemoryBackend doesn't support separate trace storage
        # Traces would need to be integrated into the main state if needed
        raise NotImplementedError("InMemoryBackend doesn't support separate trace storage")

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[JSONObject]:
        """Get individual spans with optional filtering."""
        # InMemoryBackend doesn't support normalized span storage
        return []

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> JSONObject:
        """Get aggregated span statistics."""
        # InMemoryBackend doesn't support span statistics
        return {
            "total_spans": 0,
            "by_name": {},
            "by_status": {},
            "avg_duration_by_name": {},
        }

    async def list_runs(
        self,
        status: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        metadata_filter: Optional[JSONObject] = None,
    ) -> List[JSONObject]:
        async with self._get_lock():
            entries: list[tuple[datetime, JSONObject]] = []
            for stored in self._store.values():
                state = deepcopy(stored)
                if status and state.get("status") != status:
                    continue
                if pipeline_name and state.get("pipeline_name") != pipeline_name:
                    continue
                metadata = state.get("metadata") or {}
                if metadata_filter and not metadata_contains(metadata, metadata_filter):
                    continue

                created_at = state.get("created_at")
                updated_at = state.get("updated_at")
                sort_key = self._coerce_datetime(created_at)

                entry: JSONObject = {
                    "run_id": state.get("run_id"),
                    "pipeline_name": state.get("pipeline_name"),
                    "pipeline_version": state.get("pipeline_version"),
                    "status": state.get("status"),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "metadata": metadata,
                    "start_time": created_at,
                    "end_time": updated_at,
                    "total_cost": 0.0,
                }
                entries.append((sort_key or datetime.min.replace(tzinfo=timezone.utc), entry))

            entries.sort(key=lambda item: item[0], reverse=True)
            sliced = entries[offset:]
            if limit is not None:
                sliced = sliced[:limit]
            return [entry for _, entry in sliced]

    @staticmethod
    def _coerce_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    async def set_system_state(self, key: str, value: JSONObject) -> None:
        async with self._get_lock():
            self._system_state[key] = {
                "key": key,
                "value": value,
                "updated_at": datetime.now(timezone.utc),
            }

    async def get_system_state(self, key: str) -> Optional[JSONObject]:
        async with self._get_lock():
            return self._system_state.get(key)
