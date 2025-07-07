from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StateBackend(ABC):
    """Abstract interface for workflow state persistence."""

    @abstractmethod
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Persist the serialized state for ``run_id``."""
        ...

    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load and return the serialized state for ``run_id`` if present."""
        ...

    @abstractmethod
    async def delete_state(self, run_id: str) -> None:
        """Remove any persisted state for ``run_id``."""
        ...
