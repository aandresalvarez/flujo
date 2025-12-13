from __future__ import annotations

import contextlib
import inspect
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from flujo.application.core.state_manager import StateManager
from flujo.application.runner import Flujo
from flujo.cli.config import load_backend_from_config
from flujo.client.models import SystemState, TaskDetail, TaskStatus, TaskSummary
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.infra.registry import PipelineRegistry
from flujo.state.backends.base import StateBackend
from flujo.state.models import WorkflowState
from flujo.type_definitions.common import JSONObject


class TaskClientError(RuntimeError):
    """Base error raised for TaskClient-related failures."""


class TaskNotFoundError(TaskClientError):
    """Raised when a requested run_id cannot be found in the backend."""


class TaskClient:
    """High-level facade for inspecting and resuming persisted runs."""

    def __init__(self, backend: Optional[StateBackend] = None) -> None:
        if backend is not None:
            self._backend: StateBackend = backend
            self._owns_backend = False
        else:
            self._backend = load_backend_from_config()
            self._owns_backend = True
        self._state_manager: StateManager[PipelineContext] = StateManager(self._backend)
        self._closed = False

    def __del__(self) -> None:
        # Best-effort warning for unclosed clients; avoid raising during GC.
        try:
            if self._owns_backend and not self._closed:
                import warnings

                warnings.warn(
                    "TaskClient was not closed; resources may be leaked. Use 'async with TaskClient()' or call close().",
                    ResourceWarning,
                )
        except Exception:
            pass

    @property
    def backend(self) -> StateBackend:
        """Expose the underlying backend for advanced scenarios."""
        return self._backend

    async def list_tasks(
        self,
        *,
        status: Optional[TaskStatus | str] = None,
        pipeline_name: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        metadata_filter: Optional[JSONObject] = None,
    ) -> list[TaskSummary]:
        """Return lightweight summaries for stored runs."""
        status_filter = self._normalize_status_input(status)
        try:
            runs = await self._backend.list_runs(
                status=status_filter,
                pipeline_name=pipeline_name,
                limit=limit,
                offset=offset,
                metadata_filter=metadata_filter,
            )
        except NotImplementedError as exc:  # pragma: no cover - backend contract guard
            raise TaskClientError("Configured backend does not implement list_runs") from exc

        summaries: list[TaskSummary] = []
        for run in runs:
            metadata = self._coerce_metadata(run.get("metadata"))
            created_at = self._coerce_datetime(run.get("created_at") or run.get("start_time"))
            updated_at = self._coerce_datetime(run.get("updated_at") or run.get("end_time"))
            summaries.append(
                TaskSummary(
                    run_id=str(run.get("run_id")),
                    pipeline_name=str(run.get("pipeline_name") or "unknown"),
                    pipeline_version=str(run.get("pipeline_version") or "latest"),
                    status=self._parse_status(run.get("status")),
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=metadata,
                )
            )
        return summaries

    async def get_task(
        self,
        run_id: str,
        *,
        context_model: Optional[type[PipelineContext]] = None,
    ) -> TaskDetail:
        """Return the full task detail for a specific run."""
        raw_state = await self._backend.load_state(run_id)
        if raw_state is None:
            raise TaskNotFoundError(f"Run '{run_id}' not found")

        wf_state = WorkflowState.model_validate(raw_state)
        (
            context,
            _,
            current_step_index,
            _,
            _,
            _,
            step_history,
        ) = await self._state_manager.load_workflow_state(run_id, context_model)

        context_snapshot = self._coerce_json_object(wf_state.pipeline_context)
        metadata = self._coerce_metadata(wf_state.metadata)

        return TaskDetail(
            run_id=wf_state.run_id,
            pipeline_name=wf_state.pipeline_name,
            pipeline_version=wf_state.pipeline_version,
            status=self._parse_status(wf_state.status),
            created_at=self._ensure_datetime(wf_state.created_at),
            updated_at=self._ensure_datetime(wf_state.updated_at),
            metadata=metadata,
            current_step_index=current_step_index,
            step_history=step_history,
            context_snapshot=context_snapshot,
            last_prompt=self._extract_last_prompt(context, context_snapshot, metadata),
            pending_human_input_schema=self._extract_pending_schema(context_snapshot, metadata),
            error_message=wf_state.error_message,
        )

    async def resume_task(
        self,
        run_id: str,
        pipeline_or_input: Any,
        input_data: Optional[Any] = None,
        *,
        pipeline: Optional[Pipeline[Any, Any]] = None,
        registry: Optional[PipelineRegistry] = None,
        context_model: Optional[type[PipelineContext]] = None,
    ) -> PipelineResult[Any]:
        """Resume a paused workflow run with the provided pipeline and input.

        Signature supports both old and new APIs:
        - Old: resume_task(run_id, pipeline, input_data)
        - New: resume_task(run_id, input_data, pipeline=pipeline) or resume_task(run_id, input_data, registry=registry)

        Either `pipeline` or `registry` must be provided. If `registry` is provided
        and `pipeline` is None, the pipeline will be resolved from the registry using
        the pipeline_name and pipeline_version stored in the task's metadata.
        """
        # Handle backward compatibility: old API was resume_task(run_id, pipeline, input_data)
        target_pipeline: Optional[Pipeline[Any, Any]]
        if isinstance(pipeline_or_input, Pipeline):
            # Old API: second arg is pipeline, third is input_data
            target_pipeline = pipeline_or_input
            actual_input_data = input_data
        else:
            # New API: second arg is input_data, pipeline/registry are kwargs
            target_pipeline = pipeline
            actual_input_data = pipeline_or_input if input_data is None else input_data

        paused_result = await self._state_manager.rehydrate_pipeline_result(run_id, context_model)
        if paused_result is None:
            raise TaskNotFoundError(f"Run '{run_id}' has no persisted state")

        # Resolve Pipeline
        p_name: Optional[str] = None
        if target_pipeline is None:
            if registry is None:
                raise ValueError(
                    "Must provide either 'pipeline' object or 'registry' to resume_task"
                )

            # Resolve from metadata stored in backend
            raw_state = await self._backend.load_state(run_id)
            if raw_state is None:
                raise TaskNotFoundError(f"Run '{run_id}' not found in backend")

            wf_state = WorkflowState.model_validate(raw_state)
            p_name = wf_state.pipeline_name
            p_ver = wf_state.pipeline_version or "latest"

            if not p_name:
                raise ValueError(
                    f"Run {run_id} has no recorded pipeline name to look up in registry"
                )

            target_pipeline = registry.get(p_name, p_ver)
            if target_pipeline is None:
                # Try "latest" if specific version not found
                if p_ver != "latest":
                    target_pipeline = registry.get_latest(p_name)
                if target_pipeline is None:
                    raise ValueError(f"Pipeline '{p_name}' (v={p_ver}) not found in registry")
        else:
            # If pipeline was provided, get pipeline_name from state for runner
            raw_state = await self._backend.load_state(run_id)
            if raw_state is not None:
                wf_state = WorkflowState.model_validate(raw_state)
                p_name = wf_state.pipeline_name

        runner: Flujo[Any, Any, PipelineContext] = Flujo(
            pipeline=target_pipeline,
            state_backend=self._backend,
            pipeline_name=p_name,
            enable_tracing=True,
        )
        try:
            return await runner.resume_async(paused_result, actual_input_data)
        finally:
            with contextlib.suppress(Exception):
                await runner.aclose()

    async def set_system_state(self, key: str, value: JSONObject) -> SystemState:
        """Persist a system-wide marker (e.g., connector watermark)."""
        try:
            await self._backend.set_system_state(key, value)
            stored = await self._backend.get_system_state(key)
        except NotImplementedError as exc:  # pragma: no cover - backend contract guard
            raise TaskClientError(
                "Configured backend does not implement system state storage"
            ) from exc

        if stored is None:
            return SystemState(key=key, value=value, updated_at=datetime.now(timezone.utc))
        return self._build_system_state(stored)

    async def get_system_state(self, key: str) -> Optional[SystemState]:
        """Fetch a system-wide marker by key."""
        try:
            stored = await self._backend.get_system_state(key)
        except NotImplementedError as exc:  # pragma: no cover - backend contract guard
            raise TaskClientError(
                "Configured backend does not implement system state storage"
            ) from exc

        if stored is None:
            return None
        return self._build_system_state(stored)

    async def close(self) -> None:
        """Cleanly shutdown the backend connection if owned by this client.

        If the backend was injected externally (passed into __init__), this method
        will not close it, as the caller owns the lifecycle.
        """
        if self._closed:
            return
        if not self._owns_backend:
            self._closed = True
            return

        shutdown_fn = getattr(self._backend, "shutdown", None)
        if shutdown_fn is not None and callable(shutdown_fn):
            result = shutdown_fn()
            if inspect.isawaitable(result):
                await result
            self._closed = True
            return

        close_fn = getattr(self._backend, "close", None)
        if close_fn is not None and callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result

        self._closed = True

    def _build_system_state(self, record: JSONObject) -> SystemState:
        value = self._coerce_json_object(record.get("value"))
        updated_at = self._coerce_datetime(record.get("updated_at"))
        key = str(record.get("key"))
        return SystemState(key=key, value=value, updated_at=updated_at)

    def _normalize_status_input(self, status: Optional[TaskStatus | str]) -> Optional[str]:
        if status is None:
            return None
        if isinstance(status, TaskStatus):
            return status.value
        return str(status).lower()

    def _parse_status(self, status: Any) -> TaskStatus:
        raw = str(status or "").lower()
        try:
            return TaskStatus(raw)
        except ValueError:
            return TaskStatus.UNKNOWN

    def _coerce_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                dt = datetime.fromtimestamp(0, tz=timezone.utc)
        elif value is None:
            dt = datetime.now(timezone.utc)
        else:
            dt = datetime.fromtimestamp(0, tz=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _ensure_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def _coerce_metadata(self, metadata: Any) -> JSONObject:
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _coerce_json_object(self, value: Any) -> JSONObject:
        return value if isinstance(value, dict) else {}

    def _extract_last_prompt(
        self,
        context: Optional[PipelineContext],
        context_snapshot: JSONObject,
        metadata: JSONObject,
    ) -> Optional[str]:
        if context is not None:
            try:
                val = getattr(context, "pause_message", None)
                if isinstance(val, str) and val.strip():
                    return val
            except Exception:
                pass

        for candidate in (context_snapshot, metadata):
            if isinstance(candidate, dict):
                for key in ("pause_message", "hitl_message"):
                    val = candidate.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
        return None

    def _extract_pending_schema(
        self,
        context_snapshot: JSONObject,
        metadata: JSONObject,
    ) -> Optional[JSONObject]:
        candidates: Sequence[Any] = (
            context_snapshot,
            metadata,
        )
        for candidate in candidates:
            if isinstance(candidate, dict):
                for key in (
                    "pending_human_input_schema",
                    "pending_resume_schema",
                    "hitl_schema",
                    "human_input_schema",
                ):
                    schema = candidate.get(key)
                    if isinstance(schema, dict):
                        return schema
        return None

    async def __aenter__(self) -> "TaskClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()


__all__ = ["TaskClient", "TaskClientError", "TaskNotFoundError"]
