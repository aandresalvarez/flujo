from __future__ import annotations

from datetime import datetime, UTC
from typing import Any, List, Literal

from flujo.type_definitions.common import JSONObject

from ..domain.models import BaseModel
from pydantic import Field


class WorkflowState(BaseModel):
    """Serialized snapshot of a running workflow."""

    run_id: str
    pipeline_id: str
    pipeline_name: str
    pipeline_version: str
    current_step_index: int
    pipeline_context: JSONObject
    last_step_output: Any | None = None
    step_history: List[JSONObject] = Field(default_factory=list)  # Serialized StepResult objects
    status: Literal["running", "paused", "completed", "failed", "cancelled"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
    metadata: JSONObject = Field(default_factory=dict)
    is_background_task: bool = False
    parent_run_id: str | None = None
    task_id: str | None = None
    background_error: str | None = None


__all__ = ["WorkflowState"]
