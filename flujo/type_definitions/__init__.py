"""Type definitions and aliases for Flujo.

This module provides type aliases and definitions to improve type safety
throughout the Flujo codebase, reducing reliance on `Any` and `JSONObject`.
"""

from .common import JSONObject, JSONArray
from .validation import ValidationIssue, ValidationResult

__all__ = [
    "JSONObject",
    "JSONArray",
    "ValidationIssue",
    "ValidationResult",
]
