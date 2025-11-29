"""Type definitions for validation results.

This module provides typed structures for validation results, enabling
type-safe validation and error handling throughout Flujo.
"""

from typing import List, Optional, Any, Literal
from typing_extensions import TypedDict, NotRequired


class ValidationIssue(TypedDict):
    """A single validation issue."""

    field: str
    message: str
    severity: Literal["error", "warning", "info"]


class ValidationResult(TypedDict):
    """Result of a validation operation."""

    is_valid: bool
    issues: List[ValidationIssue]
    data_quality_score: float  # 0.0 to 1.0
    sanitized_data: NotRequired[Optional[Any]]


class APIResponseValidationResult(ValidationResult):
    """Extended validation result for API responses."""

    response_code: NotRequired[int]
    raw_response: NotRequired[Any]


__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "APIResponseValidationResult",
]
