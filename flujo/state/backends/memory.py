from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, Optional, Callable

from .base import StateBackend


class InMemoryBackend(StateBackend):
    """Simple in-memory backend for testing and defaults."""

    def __init__(self, *, serializer_default: Callable[[Any], Any] | None = None) -> None:
        super().__init__(serializer_default=serializer_default)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            self._store[run_id] = copy.deepcopy(state)

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return copy.deepcopy(self._store.get(run_id))

    async def delete_state(self, run_id: str) -> None:
        async with self._lock:
            self._store.pop(run_id, None)
