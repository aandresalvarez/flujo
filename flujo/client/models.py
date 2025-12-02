from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from flujo.domain.models import BaseModel, StepResult
from flujo.type_definitions.common import JSONObject
from pydantic import Field


class TaskStatus(str, Enum):
    """Normalized status values exposed by the public client API."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class TaskSummary(BaseModel):
    """Lightweight run summary used for list views."""

    run_id: str
    pipeline_name: str
    pipeline_version: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    metadata: JSONObject = Field(default_factory=dict)


class TaskDetail(TaskSummary):
    """Full run detail including historical context."""

    current_step_index: int = 0
    step_history: List[StepResult] = Field(default_factory=list)
    context_snapshot: JSONObject = Field(default_factory=dict)
    last_prompt: Optional[str] = None
    pending_human_input_schema: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SystemState(BaseModel):
    """Represents a system-wide key/value tuple."""

    key: str
    value: JSONObject
    updated_at: datetime


__all__ = ["TaskStatus", "TaskSummary", "TaskDetail", "SystemState"]
