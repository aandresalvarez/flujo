"""Base classes for state backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


def _to_jsonable(obj: object) -> object:
    """Convert an object to a JSON-serializable format.

    This function handles Pydantic models and nested structures by converting
    them to dictionaries and lists that can be serialized to JSON.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    return obj


class StateBackend(ABC):
    """Abstract base class for state backends."""

    @abstractmethod
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Save state for a run."""
        pass

    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a run."""
        pass

    @abstractmethod
    async def delete_state(self, run_id: str) -> None:
        """Delete state for a run."""
        pass

    # Optional: Observability/admin methods (default: NotImplemented)
    async def list_workflows(
        self,
        status: Optional[str] = None,
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
