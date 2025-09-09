from __future__ import annotations

from typing import Literal, Optional, List

from pydantic import BaseModel, Field


class ValidationFinding(BaseModel):
    """Represents a single validation finding."""

    rule_id: str
    severity: Literal["error", "warning"]
    message: str
    step_name: Optional[str] = None
    suggestion: Optional[str] = None
    # Optional extended fields for richer diagnostics
    location_path: Optional[str] = None  # e.g., steps[3].input
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    # Import aggregation context (optional)
    # import_alias: alias used under parent's `imports:` mapping for immediate parent
    import_alias: Optional[str] = None
    # import_stack: outermost→innermost aliases for nested imports, e.g., ["clarification", "extract"]
    import_stack: Optional[List[str]] = None


class ValidationReport(BaseModel):
    """Aggregated validation results for a pipeline."""

    errors: List[ValidationFinding] = Field(default_factory=list)
    warnings: List[ValidationFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


__all__ = ["ValidationFinding", "ValidationReport"]
