from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Dict, Optional, cast, List, Tuple

from ...utils.serialization import safe_serialize, safe_deserialize

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
        self._store: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            # Serialize state so custom types are handled consistently
            self._store[run_id] = safe_serialize(state)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            stored = self._store.get(run_id)
            if stored is None:
                return None
            # Return a deserialized copy to avoid accidental mutation
            return deepcopy(cast(Dict[str, Any], safe_deserialize(stored)))

    async def delete_state(self, run_id: str) -> None:
        async with self._lock:
            self._store.pop(run_id, None)

    async def get_trace(self, run_id: str) -> Any:
        """Retrieve trace data for a given run_id."""
        # InMemoryBackend doesn't support separate trace storage
        return None

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get individual spans with optional filtering."""
        # InMemoryBackend doesn't support normalized span storage
        return []

    async def get_span_statistics(
        self, pipeline_name: Optional[str] = None, time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Get aggregated span statistics."""
        # InMemoryBackend doesn't support span statistics
        return {"total_spans": 0, "by_name": {}, "by_status": {}, "avg_duration_by_name": {}}
