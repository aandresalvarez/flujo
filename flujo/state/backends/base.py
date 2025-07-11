from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Literal


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

    # Optional: Observability/admin methods (default: NotImplemented)
    async def list_workflows(
        self,
        status: Optional[Literal["running", "paused", "completed", "failed", "cancelled"]] = None,
        pipeline_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List workflows with optional filtering and pagination."""
        raise NotImplementedError

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about stored workflows."""
        raise NotImplementedError

    async def cleanup_old_workflows(self, days_old: int = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        raise NotImplementedError

    async def get_failed_workflows(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get failed workflows from the last N hours with error details."""
        raise NotImplementedError
