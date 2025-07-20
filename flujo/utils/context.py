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

# Fields that should be excluded from context merging to prevent duplication
EXCLUDED_FIELDS = {"command_log"}


def safe_merge_context_updates(
    target_context: T, source_context: T, context_type: Optional[Type[T]] = None
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
                if field_name in EXCLUDED_FIELDS:
                    continue

                # Check if field exists in target
                if not hasattr(target_context, field_name):
                    continue

                # Get current value for comparison
                current_value = getattr(target_context, field_name)

                # Compare values safely
                try:
                    if current_value != source_value:
                        # Use setattr to trigger Pydantic validation
                        setattr(target_context, field_name, source_value)
                        updated_count += 1
                except (TypeError, ValueError, AttributeError, ValidationError) as e:
                    # Skip fields that can't be compared or set
                    logger.debug(f"Skipping field '{field_name}' due to comparison/set error: {e}")
                    continue

            except (AttributeError, TypeError, ValidationError) as e:
                # Skip fields that can't be accessed or set
                logger.debug(f"Skipping field '{field_name}' due to access/set error: {e}")
                continue

        if updated_count > 0:
            logger.debug(f"Successfully updated {updated_count} fields in context")

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
