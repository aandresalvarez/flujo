"""
Context utilities for Flujo framework.

This module provides utilities for working with Pydantic-based context objects,
including safe merging operations that respect Pydantic validation.
"""

from typing import Any, Optional, Type, TypeVar
import logging

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Cache for excluded fields to avoid repeated environment variable access
_EXCLUDED_FIELDS_CACHE: Optional[set[str]] = None
_ENV_EXCLUDED_FIELDS_CACHE: Optional[str] = None


def get_excluded_fields() -> set[str]:
    """
    Retrieve the set of fields to exclude from context merging.

    The default excluded fields are:
    - 'command_log': Avoids redundant or conflicting entries during loop operations.

    Returns:
        set[str]: A set of field names to exclude.
    """
    global _EXCLUDED_FIELDS_CACHE

    # Return cached result if available
    if _EXCLUDED_FIELDS_CACHE is not None:
        return _EXCLUDED_FIELDS_CACHE

    # Default excluded fields
    # 'command_log' is excluded to prevent redundant or conflicting entries during loop operations
    # where the same command might be logged multiple times across iterations
    default_excluded_fields = {"command_log"}

    # Retrieve excluded fields from configuration (e.g., environment variable or file)
    # For simplicity, this example uses an environment variable.
    import os

    # Cache the environment variable value to prevent repeated access
    global _ENV_EXCLUDED_FIELDS_CACHE
    if _ENV_EXCLUDED_FIELDS_CACHE is None:
        _ENV_EXCLUDED_FIELDS_CACHE = os.getenv("EXCLUDED_FIELDS", "")

    excluded_fields = _ENV_EXCLUDED_FIELDS_CACHE

    # Whitelist of allowed field names for security
    ALLOWED_EXCLUDED_FIELDS = {
        "command_log",
        "cache_hits",
        "cache_misses",
        "processing_history",
        "current_operation",
        "operation_count",
        "cache_timestamps",
        "cache_keys",
        "scratchpad",
        "hitl_history",
        "run_id",
        "initial_prompt",
    }

    # Maximum allowed length for field names to prevent abuse
    MAX_FIELD_NAME_LENGTH = 50

    if excluded_fields:
        # Validate and sanitize the field names against whitelist
        sanitized_fields = set()
        for field in excluded_fields.split(","):
            field = field.strip()

            # Skip empty entries
            if not field:
                continue

            # Enforce maximum length to mitigate potential DoS attacks with huge env vars
            if len(field) > MAX_FIELD_NAME_LENGTH:
                logger.warning(
                    f"Field '{field}' exceeds maximum length of {MAX_FIELD_NAME_LENGTH}. Skipping."
                )
                continue

            # Ensure field is a valid Python identifier (alphanumeric + underscores, no leading digits)
            if not field.isidentifier():
                logger.warning(f"Field '{field}' contains invalid characters. Skipping.")
                continue

            # Whitelist validation
            if field in ALLOWED_EXCLUDED_FIELDS:
                sanitized_fields.add(field)
            else:
                logger.warning(
                    f"Field '{field}' from EXCLUDED_FIELDS environment variable is not in allowed whitelist. Skipping."
                )

        if sanitized_fields:
            _EXCLUDED_FIELDS_CACHE = sanitized_fields
            return sanitized_fields
        else:
            logger.warning(
                "Environment variable 'EXCLUDED_FIELDS' contains no valid field names. Falling back to default excluded fields."
            )

    _EXCLUDED_FIELDS_CACHE = default_excluded_fields
    return default_excluded_fields


def safe_merge_context_updates(
    target_context: T,
    source_context: T,
    context_type: Optional[Type[T]] = None,
    excluded_fields: Optional[set[str]] = None,
) -> bool:
    """
    Safely merge context updates from source to target, respecting Pydantic validation.

    This function properly handles Pydantic models by:
    1. Using model_dump() to get field values (works with both v1 and v2)
    2. Using setattr() for updates to trigger validation
    3. Handling computed fields and validators properly
    4. Gracefully handling equality comparison failures

    Args:
        target_context: The target context to update
        source_context: The source context with updates
        context_type: Optional type hint for better error handling

    Returns:
        bool: True if merge was successful, False if any errors occurred
    """
    if target_context is None or source_context is None:
        return False

    # Use default excluded fields if none provided
    if excluded_fields is None:
        excluded_fields = get_excluded_fields()

    try:
        # Get field values using Pydantic's model_dump() method
        # This works for both Pydantic v1 and v2
        if hasattr(source_context, "model_dump"):
            # Pydantic v2
            source_fields = source_context.model_dump()
        elif hasattr(source_context, "dict"):
            # Pydantic v1
            source_fields = source_context.dict()
        else:
            # Fallback for non-Pydantic objects
            source_fields = {
                key: value
                for key, value in source_context.__dict__.items()
                if not key.startswith("_")
            }

        # Update only changed fields using setattr to trigger validation
        updated_count = 0
        for field_name, source_value in source_fields.items():
            try:
                # Skip private fields
                if field_name.startswith("_"):
                    continue

                # Skip excluded fields to prevent duplication during loop merging
                if field_name in excluded_fields:
                    continue

                # Check if field exists in target
                if not hasattr(target_context, field_name):
                    continue

                # Always get the actual value from the source context for merging
                actual_source_value = getattr(source_context, field_name)
                current_value = getattr(target_context, field_name)

                # Compare values safely
                try:
                    if current_value != actual_source_value:
                        # Use setattr to trigger Pydantic validation
                        setattr(target_context, field_name, actual_source_value)
                        updated_count += 1
                except (TypeError, ValueError, AttributeError, ValidationError) as e:
                    # Skip fields that can't be compared or set
                    logger.debug(
                        "Skipping field '%s' due to comparison/set error: %s", field_name, e
                    )
                    continue

            except (AttributeError, TypeError, ValidationError) as e:
                # Skip fields that can't be accessed or set
                logger.debug("Skipping field '%s' due to access/set error: %s", field_name, e)
                continue

        if updated_count > 0:
            logger.debug("Successfully updated %d fields in context", updated_count)

        return True

    except Exception as e:
        logger.error(f"Failed to merge context updates: {e}")
        return False


def safe_context_field_update(context: Any, field_name: str, new_value: Any) -> bool:
    """
    Safely update a single field in a context object, respecting Pydantic validation.

    Args:
        context: The context object to update
        field_name: Name of the field to update
        new_value: New value for the field

    Returns:
        bool: True if update was successful, False otherwise
    """
    if context is None:
        return False

    try:
        # Check if field exists
        if not hasattr(context, field_name):
            return False

        # Get current value for comparison
        current_value = getattr(context, field_name)

        # Compare values safely
        try:
            if current_value != new_value:
                # Use setattr to trigger Pydantic validation
                setattr(context, field_name, new_value)
                return True
        except (TypeError, ValueError, AttributeError, ValidationError) as e:
            logger.debug(f"Cannot update field '{field_name}': {e}")
            return False

    except (AttributeError, TypeError, ValidationError) as e:
        logger.debug(f"Cannot access or update field '{field_name}': {e}")
        return False

    return False


def get_context_field_safely(context: Any, field_name: str, default: Any = None) -> Any:
    """
    Safely get a field value from a context object.

    Args:
        context: The context object
        field_name: Name of the field to get
        default: Default value if field doesn't exist or can't be accessed

    Returns:
        The field value or default
    """
    if context is None:
        return default

    try:
        if hasattr(context, field_name):
            return getattr(context, field_name)
    except (AttributeError, TypeError):
        pass

    return default


def has_context_field(context: Any, field_name: str) -> bool:
    """
    Check if a context object has a specific field.

    Args:
        context: The context object
        field_name: Name of the field to check

    Returns:
        bool: True if field exists and is accessible
    """
    if context is None:
        return False

    try:
        return hasattr(context, field_name)
    except (AttributeError, TypeError):
        return False
