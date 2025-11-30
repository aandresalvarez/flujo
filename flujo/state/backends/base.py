"""Base classes for state backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple

from ...utils.serialization import safe_serialize


def _to_jsonable(obj: object) -> object:
    """Convert an object to a JSON-serializable format.

    This function handles Pydantic models and nested structures by converting
    them to dictionaries and lists that can be serialized to JSON.

    DEPRECATED: Use safe_serialize from flujo.utils.serialization instead.
    This function is kept for backward compatibility.
    """
    return safe_serialize(obj)


class StateBackend(ABC):
    """Abstract base class for state backends.

    State backends are responsible for persisting and retrieving workflow state.
    They handle serialization of complex objects automatically using the enhanced
    serialization utilities.
    """

    @abstractmethod
    async def save_state(self, run_id: str, state: Dict[str, Any]) -> None:
        """Save workflow state.

        Args:
            run_id: Unique identifier for the workflow run
            state: Dictionary containing workflow state data
        """
        pass

    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state.

        Args:
            run_id: Unique identifier for the workflow run

        Returns:
            Dictionary containing workflow state data, or None if not found
        """
        pass

    @abstractmethod
    async def delete_state(self, run_id: str) -> None:
        """Delete workflow state.

        Args:
            run_id: Unique identifier for the workflow run
        """
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

    async def list_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List background tasks with optional filtering and pagination."""
        try:
            workflows = await self.list_workflows(status=status, limit=limit, offset=offset)
        except Exception:
            return []

        background_tasks = [
            wf
            for wf in workflows
            if wf.get("metadata", {}).get("is_background_task") or wf.get("is_background_task")
        ]

        if parent_run_id is not None:
            background_tasks = [
                wf
                for wf in background_tasks
                if wf.get("metadata", {}).get("parent_run_id") == parent_run_id
                or wf.get("parent_run_id") == parent_run_id
            ]

        return background_tasks

    async def get_failed_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        hours_back: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get failed background tasks within a time window."""
        tasks = await self.list_background_tasks(parent_run_id=parent_run_id, status="failed")
        return tasks

    @abstractmethod
    async def get_trace(self, run_id: str) -> Any:
        """Retrieve and deserialize the trace tree for a given run_id."""
        raise NotImplementedError

    @abstractmethod
    async def save_trace(self, run_id: str, trace: Dict[str, Any]) -> None:
        """Save trace data for a given run_id.

        Args:
            run_id: Unique identifier for the workflow run
            trace: Dictionary containing trace tree data
        """
        raise NotImplementedError

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get individual spans with optional filtering."""
        raise NotImplementedError

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Get aggregated span statistics."""
        raise NotImplementedError

    async def cleanup_stale_background_tasks(self, stale_hours: int = 24) -> int:
        """Mark stale background tasks as failed (timeout)."""
        return 0

    # --- New structured persistence API ---
    async def save_run_start(self, run_data: Dict[str, Any]) -> None:
        """Persist initial run metadata."""
        raise NotImplementedError

    async def save_step_result(self, step_data: Dict[str, Any]) -> None:
        """Persist a single step execution record."""
        raise NotImplementedError

    async def save_run_end(self, run_id: str, end_data: Dict[str, Any]) -> None:
        """Update run metadata when execution finishes."""
        raise NotImplementedError

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a run."""
        raise NotImplementedError

    async def list_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        """Return all step records for a run ordered by step index."""
        raise NotImplementedError

    # Optional lifecycle hook: backends may override to release resources
    async def shutdown(self) -> None:
        """Gracefully release any resources held by the backend.

        Default is a no-op. Concrete backends should override when they hold
        threads, file handles, or async connections that need closing.
        """
        return None
