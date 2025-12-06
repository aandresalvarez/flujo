from __future__ import annotations

from typing import Optional

from .base_model import BaseModel


class EvaluationScore(BaseModel):
    """Structured evaluation returned by the shadow judge."""

    score: float
    reasoning: Optional[str] = None
    criteria: dict[str, float] | None = None
