"""Serialization utilities for flujo."""

import warnings
from typing import Any, Callable, TypeVar, Type

T = TypeVar("T")

# Global registry for custom serializers
default_serializer_registry: dict[type, Callable[[Any], Any]] = {}


def register_custom_serializer(obj_type: type, serializer_func: Callable[[Any], Any]) -> None:
    """Register a custom serializer for a specific type globally.

    This function registers a serializer that will be used by the BaseModel's
    fallback serialization mechanism.
    """
    default_serializer_registry[obj_type] = serializer_func


def lookup_custom_serializer(value: Any) -> Callable[[Any], Any] | None:
    """Return a registered serializer for the value's type, or None if not found."""
    for typ, func in default_serializer_registry.items():
        if isinstance(value, typ):
            return func
    return None


def create_field_serializer(
    field_name: str, serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Create a field_serializer method for a specific field.

    This is a helper function that creates a field_serializer method that can be used
    within a Pydantic model class.

    Example:
        class MyModel(BaseModel):
            complex_object: ComplexType

            @field_serializer('complex_object', when_used='json')
            def serialize_complex_object(self, value: ComplexType) -> dict:
                return create_field_serializer('complex_object', lambda x: x.to_dict())(value)
    """

    def serialize_field(value: Any) -> Any:
        return serializer_func(value)

    return serialize_field


def serializable_field(serializer_func: Callable[[Any], Any]) -> Callable[[T], T]:
    """Decorator to mark a field as serializable with a custom serializer.

    DEPRECATED: This function has fundamental design issues with Pydantic v2.
    Use register_custom_serializer() for global serialization or create field_serializer
    methods manually for field-specific serialization.

    This function was intended to work as a field decorator but doesn't integrate
    properly with Pydantic's field_serializer system.

    Recommended alternatives:
    1. Global registry: register_custom_serializer(MyType, lambda x: x.to_dict())
    2. Manual field_serializer: @field_serializer('field_name', when_used='json')
    3. Use the global registry with model_dump(mode="json")

    Example of proper usage:
        # Instead of @serializable_field(lambda x: x.to_dict())
        # Use the global registry:
        register_custom_serializer(ComplexType, lambda x: x.to_dict())

        class MyModel(BaseModel):
            complex_object: ComplexType  # Will use global serializer
    """
    warnings.warn(
        "serializable_field is deprecated due to fundamental design issues with Pydantic v2. "
        "Use register_custom_serializer() for global serialization or create field_serializer "
        "methods manually for field-specific serialization.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorator(field: T) -> T:
        # This doesn't actually work with Pydantic v2, but we'll try to provide
        # some basic functionality for backward compatibility
        field_name = getattr(field, "__name__", None)
        if field_name is None:
            raise ValueError(
                "serializable_field must be used as a field decorator, not a class decorator. "
                "However, this approach is deprecated. Use register_custom_serializer() instead."
            )

        # Store the serializer function for potential later use
        setattr(field, "_serializer_func", serializer_func)

        return field

    return decorator


def create_serializer_for_type(
    obj_type: Type[Any], serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Create a serializer function that handles a specific type.

    This is useful for creating custom serializers that can be used with
    the BaseModel's fallback serialization.

    Example:
        def serialize_my_type(obj):
            return {"data": obj.data, "metadata": obj.metadata}

        # Register the serializer
        MyTypeSerializer = create_serializer_for_type(MyType, serialize_my_type)
    """

    def serializer(value: Any) -> Any:
        if isinstance(value, obj_type):
            return serializer_func(value)
        return value

    return serializer


def safe_serialize(obj: Any, default_serializer: Callable[[Any], Any] = str) -> Any:
    """Safely serialize an object with intelligent fallback handling.

    This function attempts to serialize an object using Pydantic's native
    serialization, and falls back to a custom serializer if that fails.

    Args:
        obj: The object to serialize
        default_serializer: Function to use as fallback (default: str)

    Returns:
        Serialized representation of the object
    """
    if obj is None:
        return None

    # Try Pydantic serialization first
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass

    # Try global registry for custom types
    func = lookup_custom_serializer(obj)
    if func:
        try:
            return func(obj)
        except Exception:
            pass

    # Handle nested structures recursively
    if isinstance(obj, dict):
        try:
            return {k: safe_serialize(v, default_serializer) for k, v in obj.items()}
        except Exception:
            pass
    elif isinstance(obj, (list, tuple, set, frozenset)):
        try:
            return [safe_serialize(v, default_serializer) for v in obj]
        except Exception:
            pass

    # Try JSON serialization
    try:
        import json

        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # Fallback to default serializer with circular reference protection
    try:
        return default_serializer(obj)
    except (RecursionError, Exception):
        # Handle circular references and other serialization failures
        return f"<{type(obj).__name__} circular>"


def _serialize_for_key(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects to cacheable structures for cache keys and robust serialization."""
    if obj is None:
        return None
    from flujo.domain.models import PipelineContext

    # Special handling for PipelineContext: exclude run_id
    if isinstance(obj, PipelineContext):
        try:
            d = obj.model_dump(mode="json")
            d.pop("run_id", None)
            return {k: _serialize_for_key(v) for k, v in d.items()}
        except (ValueError, RecursionError):
            return f"<{type(obj).__name__} circular>"
    # Special handling for Step: serialize agent by class name
    if "Step" in str(type(obj)):
        try:
            d = obj.model_dump(mode="json")
            if "agent" in d and d["agent"] is not None:
                # Get the original agent object, not the serialized value
                original_agent = getattr(obj, "agent", None)
                if original_agent is not None:
                    d["agent"] = type(original_agent).__name__
            return {k: _serialize_for_key(v) for k, v in d.items()}
        except (ValueError, RecursionError):
            return f"<{type(obj).__name__} circular>"
    # Always check BaseModel before list/tuple/set
    if hasattr(obj, "model_dump"):
        try:
            d = obj.model_dump(mode="json")
            if "run_id" in d:
                d.pop("run_id", None)
            result = {}
            for k, v in d.items():
                orig_v = getattr(obj, k, None)
                if isinstance(v, str) and isinstance(orig_v, dict):
                    # Serialize the original dict value directly (not wrapped)
                    result[k] = {kk: _serialize_for_key(vv) for kk, vv in orig_v.items()}
                else:
                    result[k] = _serialize_for_key(v)
            return result
        except (ValueError, RecursionError):
            # If model_dump fails, try to serialize manually by checking for custom serializers
            try:
                result_dict_fallback = {}
                for field_name, field_value in obj.__dict__.items():
                    if field_name.startswith("_"):
                        continue
                    func = lookup_custom_serializer(field_value)
                    if func:
                        try:
                            serialized_value = func(field_value)
                            result_dict_fallback[field_name] = serialized_value
                        except Exception:
                            result_dict_fallback[field_name] = _serialize_for_key(field_value)
                    else:
                        result_dict_fallback[field_name] = _serialize_for_key(field_value)
                return result_dict_fallback
            except Exception:
                return f"<{type(obj).__name__} circular>"
    if isinstance(obj, dict):
        d = dict(obj)
        if "run_id" in d and "initial_prompt" in d:
            d.pop("run_id", None)
        result_dict: dict[Any, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, (list, tuple, set, frozenset)):
                result_dict[k] = _serialize_for_key(v)
            else:
                func = lookup_custom_serializer(v)
                if func:
                    try:
                        serialized_value = func(v)
                        result_dict[k] = serialized_value
                        continue
                    except Exception:
                        pass
                result_dict[k] = _serialize_for_key(v)
        return result_dict
    if isinstance(obj, (list, tuple, set, frozenset)):
        result_list = []
        for v in obj:
            if isinstance(v, dict) or isinstance(v, (list, tuple, set, frozenset)):
                result_list.append(_serialize_for_key(v))
            else:
                func = lookup_custom_serializer(v)
                if func:
                    try:
                        serialized_value = func(v)
                        result_list.append(serialized_value)
                        continue
                    except Exception:
                        pass
                result_list.append(_serialize_for_key(v))
        return result_list
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', '<unknown>')}.{getattr(obj, '__qualname__', repr(obj))}"
        )
    # Handle strings that look like lists of model instances
    if isinstance(obj, str) and ("Model(" in obj or "Context(" in obj) and "run_id=" in obj:
        import re

        result_str = re.sub(r"run_id='[^']*',?\s*", "", obj)
        result_str = re.sub(r",\s*\)", ")", result_str)
        return result_str
    # --- Check global custom serializer registry for any other type ---
    func = lookup_custom_serializer(obj)
    if func:
        try:
            return _serialize_for_key(func(obj))
        except Exception:
            pass
    try:
        import json

        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _serialize_list_for_key(obj_list: list[Any]) -> list[Any]:
    """Helper function to serialize lists, ensuring BaseModel instances are converted to dicts."""
    result_list: list[Any] = []
    for v in obj_list:
        if hasattr(v, "model_dump"):
            d = v.model_dump(mode="json")
            if "run_id" in d:
                d.pop("run_id", None)
            result_list.append({k: _serialize_for_key(val) for k, val in d.items()})
        elif isinstance(v, dict):
            result_list.append({k: _serialize_for_key(val) for k, val in v.items()})
        elif isinstance(v, (list, tuple, set, frozenset)):
            result_list.append(_serialize_list_for_key(list(v)))
        else:
            result_list.append(_serialize_for_key(v))
    return result_list
