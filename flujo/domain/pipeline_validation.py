from __future__ import annotations

from typing import Literal, Optional, List

from pydantic import BaseModel, Field


class ValidationFinding(BaseModel):
    """Represents a single validation finding."""

    rule_id: str
    severity: Literal["error", "warning"]
    message: str
    step_name: Optional[str] = None


class ValidationReport(BaseModel):
    """Aggregated validation results for a pipeline."""

    errors: List[ValidationFinding] = Field(default_factory=list)
    warnings: List[ValidationFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


__all__ = ["ValidationFinding", "ValidationReport"]
