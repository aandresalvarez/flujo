"""Public facade for Flujo type definitions.

This module provides stable aliases to improve type safety throughout the
Flujo codebase, reducing reliance on `Any` and `JSONObject`.
"""

from .common import JSONObject, JSONArray, JSONValue, JSONPrimitive
from .validation import ValidationIssue, ValidationResult

__all__ = [
    "JSONObject",
    "JSONArray",
    "JSONValue",
    "JSONPrimitive",
    "ValidationIssue",
    "ValidationResult",
]
