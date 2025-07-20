"""Context utilities for Flujo framework."""

from typing import Any, Set


def get_context_fields(context: Any) -> Set[str]:
    """Get context fields with Pydantic v1/v2 compatibility."""
    if hasattr(context, "model_fields"):
        return set(context.model_fields.keys())
    elif hasattr(context, "__fields__"):
        return set(context.__fields__.keys())
    else:
        return set()


def validate_context_fields(context: Any, field_names: Set[str]) -> Set[str]:
    """Validate that field names exist in the context and return valid ones."""
    context_fields = get_context_fields(context)
    return field_names.intersection(context_fields)


def get_invalid_fields(context: Any, field_names: Set[str]) -> Set[str]:
    """Get field names that don't exist in the context."""
    context_fields = get_context_fields(context)
    return field_names - context_fields
