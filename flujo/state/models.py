from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal

from ..domain.models import BaseModel
from pydantic import Field


class WorkflowState(BaseModel):
    """Serialized snapshot of a running workflow."""

    run_id: str
    pipeline_id: str
    pipeline_name: str
    pipeline_version: str
    current_step_index: int
    pipeline_context: Dict[str, Any]
    last_step_output: Any | None = None
    step_history: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Serialized StepResult objects
    status: Literal["running", "paused", "completed", "failed", "cancelled"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = ["WorkflowState"]
