from __future__ import annotations

import types
import re
from typing import Any, Optional, Type, Union, Dict, TypeVar, cast

from pydantic import ValidationError
from pydantic import BaseModel as PydanticBaseModel

from ...infra import telemetry
from ...domain.models import BaseModel

__all__ = ["_build_context_update", "_inject_context", "register_custom_type"]

# Type registry for efficient type resolution
_TYPE_REGISTRY: Dict[str, Type[Any]] = {}
T = TypeVar("T")


def _register_type(type_name: str, type_class: Type[T]) -> None:
    """Register a type in the global type registry for efficient resolution."""
    _TYPE_REGISTRY[type_name] = type_class

    # Also register common aliases for better discovery
    if hasattr(type_class, "__name__"):
        _TYPE_REGISTRY[type_class.__name__] = type_class


def register_custom_type(type_class: Type[T]) -> None:
    """Public API for users to register their custom types."""
    if hasattr(type_class, "__name__"):
        _register_type(type_class.__name__, type_class)


def _resolve_type_from_string(type_str: str) -> Optional[Type[Any]]:
    """
    Efficiently resolve a type from its string representation.

    This replaces the fragile sys.modules iteration with a deterministic,
    performant type registry approach.
    """
    # First check the type registry
    if type_str in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[type_str]

    # For backward compatibility, try to resolve from common test modules
    # This is much more targeted than iterating through all sys.modules
    try:
        # Check if it's a test-specific type that might be in the current module
        import inspect

        frame = inspect.currentframe()
        depth = 0
        max_depth = 10  # Reasonable limit for call stack depth

        while frame and depth < max_depth:
            if type_str in frame.f_globals:
                return cast(Type[Any], frame.f_globals[type_str])
            frame = frame.f_back
            depth += 1
    except Exception:
        pass

    return None


def _extract_union_types(union_type: Any) -> list[Type[Any]]:
    """
    Extract non-None types from a Union type annotation.

    Handles both old-style Union[T, None] and new-style T | None syntax.
    """
    non_none_types: list[Type[Any]] = []

    # Handle Python 3.10+ Union syntax (types.UnionType)
    if isinstance(union_type, types.UnionType):
        # Use get_args to extract the actual types instead of string parsing
        try:
            from typing import get_args

            args = get_args(union_type)
            non_none_types = [t for t in args if t is not type(None)]
        except Exception:
            # Generic fallback: extract type names from string representation
            type_str = str(union_type)
            # Use regex to find potential type names in the string
            # Match potential type names (capitalized words that could be types)
            type_pattern = r"\b[A-Z][a-zA-Z0-9_]*\b"
            potential_types = re.findall(type_pattern, type_str)

            for type_name in potential_types:
                # Skip common built-in types and generic markers
                if type_name not in ["Union", "Optional", "List", "Dict", "Any", "None"]:
                    resolved_type = _resolve_type_from_string(type_name)
                    if resolved_type:
                        non_none_types.append(resolved_type)
        return non_none_types

    # Handle traditional Union[T, None] syntax
    if hasattr(union_type, "__origin__") and union_type.__origin__ is Union:
        non_none_types = [t for t in union_type.__args__ if t is not type(None)]
    elif hasattr(union_type, "__union_params__"):
        non_none_types = [t for t in union_type.__union_params__ if t is not type(None)]

    return non_none_types


def _resolve_actual_type(field_type: Any) -> Optional[Type[Any]]:
    """
    Resolve the actual type from a field annotation, handling Union types.

    This is a robust replacement for the fragile type resolution logic.
    """
    if field_type is None:
        return None

    # Handle Union types
    if isinstance(field_type, types.UnionType) or (
        hasattr(field_type, "__origin__") and field_type.__origin__ is Union
    ):
        non_none_types = _extract_union_types(field_type)
        return non_none_types[0] if non_none_types else None

    return cast(Type[Any], field_type)


def _deserialize_value(value: Any, field_type: Any, context_model: Type[BaseModel]) -> Any:
    """
    Deserialize a value according to its field type.

    This centralizes the deserialization logic and eliminates code duplication.
    """
    if field_type is None or not isinstance(value, (dict, list)):
        return value

    # Handle list types
    if (
        isinstance(value, list)
        and hasattr(field_type, "__origin__")
        and field_type.__origin__ is list
    ):
        # Get the element type from list[T]
        element_type = field_type.__args__[0] if field_type.__args__ else None
        if element_type is not None:
            # Resolve the actual element type (handle Union types)
            actual_element_type = _resolve_actual_type(element_type)
            if actual_element_type is not None:
                # Handle Pydantic models in list
                if hasattr(actual_element_type, "model_validate") and issubclass(
                    actual_element_type, BaseModel
                ):
                    try:
                        return [
                            actual_element_type.model_validate(item)
                            if isinstance(item, dict)
                            else item
                            for item in value
                        ]
                    except Exception:
                        pass
                else:
                    # Handle custom deserializers for list elements
                    from flujo.utils.serialization import lookup_custom_deserializer

                    custom_deserializer = lookup_custom_deserializer(actual_element_type)
                    if custom_deserializer:
                        try:
                            return [custom_deserializer(item) for item in value]
                        except Exception:
                            pass
        return value

    # Handle dict types
    if isinstance(value, dict):
        actual_type = _resolve_actual_type(field_type)
        if actual_type is None:
            return value

        # Handle Pydantic models
        if hasattr(actual_type, "model_validate") and issubclass(actual_type, BaseModel):
            try:
                return actual_type.model_validate(value)
            except Exception:
                pass
        else:
            # Handle custom deserializers
            from flujo.utils.serialization import lookup_custom_deserializer

            custom_deserializer = lookup_custom_deserializer(actual_type)
            if custom_deserializer:
                try:
                    return custom_deserializer(value)
                except Exception:
                    pass

    return value


def _build_context_update(output: BaseModel | dict[str, Any] | Any) -> dict[str, Any] | None:
    """Return context update dict extracted from a step output."""
    if isinstance(output, (BaseModel, PydanticBaseModel)):
        return output.model_dump(exclude_unset=True)
    if isinstance(output, dict):
        return output
    return None


def _inject_context(
    context: BaseModel,
    update_data: dict[str, Any],
    context_model: Type[BaseModel],
) -> Optional[str]:
    """Apply ``update_data`` to ``context`` validating against ``context_model``.

    Returns an error message if validation fails, otherwise ``None``.
    """
    original = context.model_dump()

    # Generic type registration: extract all potential types from field annotations
    for field_name, field_info in context_model.model_fields.items():
        if field_info.annotation:
            type_str = str(field_info.annotation)
            # Extract potential type names using regex
            type_pattern = r"\b[A-Z][a-zA-Z0-9_]*\b"
            potential_types = re.findall(type_pattern, type_str)

            for type_name in potential_types:
                # Skip common built-in types
                if type_name not in [
                    "Union",
                    "Optional",
                    "List",
                    "Dict",
                    "Any",
                    "None",
                    "BaseModel",
                ]:
                    try:
                        import inspect

                        frame = inspect.currentframe()
                        depth = 0
                        max_depth = 10

                        while frame and depth < max_depth:
                            if type_name in frame.f_globals:
                                _register_type(type_name, frame.f_globals[type_name])
                                break
                            frame = frame.f_back
                            depth += 1
                    except Exception:
                        pass

    # Process update data
    for key, value in update_data.items():
        if key in context_model.model_fields:
            field_info = context_model.model_fields[key]
            field_type = field_info.annotation

            # Deserialize the value using the robust deserialization logic
            deserialized_value = _deserialize_value(value, field_type, context_model)
            setattr(context, key, deserialized_value)
        elif not hasattr(context, key):
            # Enhanced error handling with better messages
            from flujo.exceptions import ContextFieldError

            available_fields = (
                list(context.__fields__.keys()) if hasattr(context, "__fields__") else []
            )
            if hasattr(context, "model_fields"):
                available_fields = list(context.model_fields.keys())
            raise ContextFieldError(key, context.__class__.__name__, available_fields)
        else:
            setattr(context, key, value)

    # Final pass: re-apply deserialization to ensure consistency
    for key in context_model.model_fields:
        field_info = context_model.model_fields[key]
        field_type = field_info.annotation
        current_value = getattr(context, key, None)

        if current_value is not None:
            deserialized_value = _deserialize_value(current_value, field_type, context_model)
            if deserialized_value != current_value:
                setattr(context, key, deserialized_value)

    # CRITICAL FIX: Apply Pydantic validation results back to the context
    # This ensures normalization, coercion, and default values are properly applied
    try:
        # Validate the current context state and apply the validated results
        validated = context_model.model_validate(context.model_dump())
        # Apply the validated results back to the context to ensure normalization
        context.__dict__.update(validated.__dict__)
    except ValidationError as e:
        # If validation fails, restore original state
        for key, value in original.items():
            setattr(context, key, value)
        telemetry.logfire.error(f"Context update failed Pydantic validation: {e}")
        return str(e)
    return None
