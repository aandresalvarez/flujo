from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

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
    status: Literal["running", "paused", "completed", "failed", "cancelled"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


__all__ = ["WorkflowState"]
