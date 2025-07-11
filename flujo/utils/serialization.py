"""Serialization utilities for flujo."""

from typing import Any, Callable, TypeVar, Type
from pydantic import field_serializer
import inspect

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


def serializable_field(serializer_func: Callable[[Any], Any]) -> Callable[[T], T]:
    """Decorator to mark a field as serializable with a custom serializer.

    This is a convenience decorator that creates a @field_serializer for a specific field.

    Example:
        class MyModel(BaseModel):
            @serializable_field(lambda x: x.to_dict())
            complex_object: ComplexType
    """

    def decorator(cls: T) -> T:
        # Get the field name from the class
        field_name = None
        for name, value in inspect.getmembers(cls):
            if value is serializer_func:
                field_name = name
                break

        if field_name:
            # Create a field_serializer for this field
            @field_serializer(field_name, when_used="json")
            def serialize_field(value: Any) -> Any:
                return serializer_func(value)

            # Attach the serializer to the class
            setattr(cls, f"_serialize_{field_name}", serialize_field)

        return cls

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
    """Safely serialize an object with fallback handling.

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
            return obj.model_dump(mode="python")
        except Exception:
            pass

    # Try JSON serialization
    try:
        import json

        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    # Try global registry
    func = lookup_custom_serializer(obj)
    if func:
        try:
            return func(obj)
        except Exception:
            pass

    # Fallback to default serializer
    try:
        return default_serializer(obj)
    except Exception:
        return repr(obj)
