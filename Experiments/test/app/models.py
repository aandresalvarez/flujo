from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CandidatePlan(BaseModel):
    """Single candidate emitted by the planning agent."""

    id: str
    summary: str
    queries: List[str] = Field(default_factory=list)
    badges: Dict[str, object] = Field(default_factory=dict)


class PlanResponse(BaseModel):
    """Normalized agent response containing candidate options."""

    goal: Optional[str] = None
    candidates: List[CandidatePlan] = Field(default_factory=list)
