"""
Agent utilities for shared functionality.

This module contains utility functions shared between different agent modules
to avoid circular imports.
"""


def get_raw_output_from_exception(exc: Exception) -> str:
    """Best-effort extraction of raw output from validation-related exceptions."""
    if hasattr(exc, "message"):
        msg = getattr(exc, "message")
        if isinstance(msg, str):
            return msg
    if exc.args:
        first = exc.args[0]
        if isinstance(first, str):
            return first
    return str(exc)
