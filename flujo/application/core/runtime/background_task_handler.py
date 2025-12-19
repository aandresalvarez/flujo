from __future__ import annotations

from typing import TYPE_CHECKING

from ....domain.models import BaseModel as DomainBaseModel
from ....domain.models import Quota
from ....infra.settings import get_settings
from ....type_definitions.common import JSONObject

if TYPE_CHECKING:
    from ..state.state_manager import StateManager


class BackgroundTaskHandler:
    """Handles persistence and quota logic for background tasks (extracted from ExecutorCore)."""

    def __init__(self, state_manager: StateManager[DomainBaseModel] | None) -> None:
        self.state_manager = state_manager

    def get_background_quota(self, parent_quota: Quota | None = None) -> Quota | None:
        """Compute quota for background tasks with parent-first split."""
        settings = get_settings()
        bg_settings = getattr(settings, "background_tasks", None)
        if bg_settings is None or not bool(getattr(bg_settings, "enable_quota", False)):
            return parent_quota

        if parent_quota is not None:
            try:
                return parent_quota.split(1)[0]
            except Exception:
                try:
                    from ....infra import telemetry as _telemetry

                    _telemetry.logfire.warning(
                        "Cannot split parent quota for background task; quota disabled for task"
                    )
                except Exception:
                    pass
                return None

        try:
            return Quota(
                float(getattr(bg_settings, "max_cost_per_task", 0.0)),
                int(getattr(bg_settings, "max_tokens_per_task", 0)),
            )
        except Exception:
            return None

    async def register_background_task(
        self,
        *,
        task_id: str,
        bg_run_id: str,
        parent_run_id: str | None,
        step_name: str,
        data: object,
        context: DomainBaseModel | None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Persist initial state for a background task."""
        if self.state_manager is None:
            return
        if context is not None and not isinstance(context, DomainBaseModel):
            return

        meta = dict(metadata or {})
        meta.setdefault("is_background_task", True)
        meta.setdefault("task_id", task_id)
        meta.setdefault("parent_run_id", parent_run_id)
        meta.setdefault("step_name", step_name)
        meta.setdefault("input_data", data)

        await self.state_manager.persist_workflow_state(
            run_id=bg_run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="running",
            metadata=meta,
        )

    async def mark_background_task_completed(
        self,
        *,
        task_id: str,
        context: DomainBaseModel | None,
        metadata: JSONObject | None = None,
    ) -> None:
        """Mark a background task as completed."""
        if self.state_manager is None:
            return
        if context is None or not isinstance(context, DomainBaseModel):
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=1,
            last_step_output=None,
            status="completed",
            metadata=meta,
        )
        try:
            from ....infra import telemetry as _telemetry

            _telemetry.logfire.info(f"Background task '{task_id}' completed successfully")
        except Exception:
            pass

    async def mark_background_task_failed(
        self,
        *,
        task_id: str,
        context: DomainBaseModel | None,
        error: Exception,
        metadata: JSONObject | None = None,
    ) -> None:
        """Mark a background task as failed."""
        if self.state_manager is None:
            return
        if context is None or not isinstance(context, DomainBaseModel):
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)
        meta["background_error"] = meta.get("background_error") or str(error)

        if context is not None:
            try:
                if hasattr(context, "background_error"):
                    context.background_error = str(error)
            except Exception:
                pass

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="failed",
            metadata=meta,
        )
        try:
            from ....infra import telemetry as _telemetry

            _telemetry.logfire.error(
                f"Background task '{task_id}' failed", extra={"error": str(error)}
            )
        except Exception:
            pass

    async def mark_background_task_paused(
        self,
        *,
        task_id: str,
        context: DomainBaseModel | None,
        error: Exception,
        metadata: JSONObject | None = None,
    ) -> None:
        """Mark a background task as paused (control-flow signal)."""
        if self.state_manager is None:
            return
        if context is None or not isinstance(context, DomainBaseModel):
            return

        run_id = getattr(context, "run_id", None) if context is not None else None
        if run_id is None:
            return

        meta = dict(metadata or {})
        meta.setdefault("task_id", task_id)
        meta.setdefault("is_background_task", True)
        meta["background_error"] = meta.get("background_error") or str(error)

        await self.state_manager.persist_workflow_state(
            run_id=run_id,
            context=context,
            current_step_index=0,
            last_step_output=None,
            status="paused",
            metadata=meta,
        )
        try:
            from ....infra import telemetry as _telemetry

            _telemetry.logfire.info(
                f"Background task '{task_id}' paused", extra={"reason": str(error)}
            )
        except Exception:
            pass
