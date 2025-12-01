from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, List, Optional, Tuple, cast

from flujo.type_definitions.common import JSONObject
from ...utils.serialization import safe_deserialize, safe_serialize

from .base import StateBackend


class InMemoryBackend(StateBackend):
    """Simple in-memory backend for testing and defaults.

    This backend mirrors the serialization logic of the persistent backends by
    storing a serialized copy of the workflow state. Values are serialized with
    ``safe_serialize`` on save and reconstructed with ``safe_deserialize`` when
    loaded.
    """

    def __init__(self) -> None:
        # Store serialized copies to mimic persistent backends
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, run_id: str, state: JSONObject) -> None:
        async with self._lock:
            # Serialize state so custom types are handled consistently
            self._store[run_id] = safe_serialize(state)

    async def load_state(self, run_id: str) -> Optional[JSONObject]:
        async with self._lock:
            stored = self._store.get(run_id)
            if stored is None:
                return None
            # Return a deserialized copy to avoid accidental mutation
            return deepcopy(cast(JSONObject, safe_deserialize(stored)))

    async def delete_state(self, run_id: str) -> None:
        async with self._lock:
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
