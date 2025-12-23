"""Base classes for state backends."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Set, Tuple

from flujo.exceptions import ControlFlowError
from flujo.type_definitions.common import JSONObject
from flujo.utils.serialization import _serialize_for_json as _serialize_for_json_utils


def _serialize_for_json(
    obj: object, *, _seen: Optional[Set[int]] = None, strict: bool = True
) -> object:
    """Convert an object to a JSON-serializable format for persistence.

    This is the persistence-facing serialization entrypoint used by state backends.
    It delegates to the shared serializer in `flujo.utils.serialization`, while
    preserving the legacy strict/non-strict behavior expected by tests and
    `flujo.domain.base_model.BaseModel`.
    """
    circular_ref_placeholder: str | None = "<circular-ref>" if strict else None
    try:
        return _serialize_for_json_utils(
            obj,
            circular_ref_placeholder=circular_ref_placeholder,
            strict=strict,
            bytes_mode="utf8",
            allow_object_dict=not strict,
            _seen=_seen,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort non-strict path
        if isinstance(exc, ControlFlowError):
            raise
        if strict:
            raise
        # Non-strict persistence must not raise, but persistence backends must not
        # depend on the global robust serializer helper (guardrail). Fall back to a
        # minimal JSON-safe representation.
        try:
            if hasattr(obj, "__dict__"):
                seen = set(_seen) if _seen is not None else set()
                seen.add(id(obj))
                return _serialize_for_json_utils(
                    vars(obj),
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=False,
                    bytes_mode="utf8",
                    allow_object_dict=True,
                    _seen=seen,
                )
        except Exception:
            pass
        try:
            return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"


def _to_jsonable(obj: object) -> object:
    """Convert an object to a JSON-serializable format (backward-compatible shim)."""
    return _serialize_for_json(obj)


class StateBackend(ABC):
    """Abstract base class for state backends.

    State backends are responsible for persisting and retrieving workflow state.
    They handle serialization of complex objects automatically using the enhanced
    serialization utilities.
    """

    @abstractmethod
    async def save_state(self, run_id: str, state: JSONObject) -> None:
        """Save workflow state.

        Args:
            run_id: Unique identifier for the workflow run
            state: Dictionary containing workflow state data
        """
        pass

    @abstractmethod
    async def load_state(self, run_id: str) -> Optional[JSONObject]:
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
    ) -> List[JSONObject]:
        """List workflows with optional filtering and pagination."""
        raise NotImplementedError

    async def get_workflow_stats(self) -> JSONObject:
        """Get statistics about stored workflows."""
        raise NotImplementedError

    async def cleanup_old_workflows(self, days_old: int = 30) -> int:
        """Delete workflows older than specified days. Returns number of deleted workflows."""
        raise NotImplementedError

    async def get_failed_workflows(self, hours_back: int = 24) -> List[JSONObject]:
        """Get failed workflows from the last N hours with error details."""
        raise NotImplementedError

    async def list_background_tasks(
        self,
        parent_run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[JSONObject]:
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
    ) -> List[JSONObject]:
        """Get failed background tasks within a time window."""
        tasks = await self.list_background_tasks(parent_run_id=parent_run_id, status="failed")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        def _parse_timestamp(value: Any) -> Optional[datetime]:
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value)
                except Exception:
                    return None
            else:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        filtered: List[JSONObject] = []
        for task in tasks:
            ts_raw = task.get("created_at") or task.get("started_at")
            parsed_ts = _parse_timestamp(ts_raw)
            if parsed_ts is None:
                continue
            if parsed_ts >= cutoff:
                filtered.append(task)

        return filtered

    @abstractmethod
    async def get_trace(self, run_id: str) -> Any:
        """Retrieve and deserialize the trace tree for a given run_id."""
        raise NotImplementedError

    @abstractmethod
    async def save_trace(self, run_id: str, trace: JSONObject) -> None:
        """Save trace data for a given run_id.

        Args:
            run_id: Unique identifier for the workflow run
            trace: Dictionary containing trace tree data
        """
        raise NotImplementedError

    async def save_spans(self, run_id: str, spans: list[JSONObject]) -> None:
        """Persist normalized spans for a run.

        Args:
            run_id: Unique identifier for the workflow run
            spans: List of span dictionaries (span_id, parent_span_id, name, etc.)
        """
        raise NotImplementedError

    async def get_spans(
        self, run_id: str, status: Optional[str] = None, name: Optional[str] = None
    ) -> List[JSONObject]:
        """Get individual spans with optional filtering."""
        raise NotImplementedError

    async def get_span_statistics(
        self,
        pipeline_name: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> JSONObject:
        """Get aggregated span statistics."""
        raise NotImplementedError

    async def cleanup_stale_background_tasks(self, stale_hours: int = 24) -> int:
        """Mark stale background tasks as failed (timeout)."""
        return 0

    # --- Shadow evaluation persistence ---
    async def persist_evaluation(
        self,
        run_id: str,
        score: float,
        feedback: str | None = None,
        step_name: str | None = None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Persist shadow evaluation result (default: no-op)."""
        raise NotImplementedError

    async def list_evaluations(
        self,
        limit: int = 20,
        run_id: str | None = None,
    ) -> list[JSONObject]:
        """List persisted shadow evaluation results (default: not implemented)."""
        raise NotImplementedError

    # --- New structured persistence API ---
    async def save_run_start(self, run_data: JSONObject) -> None:
        """Persist initial run metadata."""
        raise NotImplementedError

    async def save_step_result(self, step_data: JSONObject) -> None:
        """Persist a single step execution record."""
        raise NotImplementedError

    async def save_run_end(self, run_id: str, end_data: JSONObject) -> None:
        """Update run metadata when execution finishes."""
        raise NotImplementedError

    async def get_run_details(self, run_id: str) -> Optional[JSONObject]:
        """Retrieve stored metadata for a run."""
        raise NotImplementedError

    async def list_runs(
        self,
        status: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        metadata_filter: Optional[JSONObject] = None,
    ) -> List[JSONObject]:
        """Return stored runs with optional filtering."""
        raise NotImplementedError

    async def list_run_steps(self, run_id: str) -> List[JSONObject]:
        """Return all step records for a run ordered by step index."""
        raise NotImplementedError

    async def set_system_state(self, key: str, value: JSONObject) -> None:
        """Upsert a global key/value pair used for connector watermarks."""
        raise NotImplementedError

    async def get_system_state(self, key: str) -> Optional[JSONObject]:
        """Fetch a previously stored global key/value pair."""
        raise NotImplementedError

    # Optional lifecycle hook: backends may override to release resources
    async def shutdown(self) -> None:
        """Gracefully release any resources held by the backend.

        Default is a no-op. Concrete backends should override when they hold
        threads, file handles, or async connections that need closing.
        """
        return None
