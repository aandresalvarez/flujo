from __future__ import annotations

from typing import Any, Dict, Optional

from .base import StateBackend


class InMemoryBackend(StateBackend):
    """Simple in-memory backend for testing and defaults."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        self._store[run_id] = state

    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(run_id)

    async def delete_state(self, run_id: str) -> None:
        self._store.pop(run_id, None)
