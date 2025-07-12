"""Serialization utilities for Flujo."""

import dataclasses
import json
import math
import threading
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Type, TypeVar

# Try to import Pydantic BaseModel for proper type checking
try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    BaseModel = None  # type: ignore
    HAS_PYDANTIC = False

# Global registry for custom serializers
_custom_serializers: Dict[Type[Any], Callable[[Any], Any]] = {}
# Global registry for custom deserializers
_custom_deserializers: Dict[Type[Any], Callable[[Any], Any]] = {}
_registry_lock = threading.Lock()

T = TypeVar("T")


def register_custom_serializer(obj_type: Type[Any], serializer_func: Callable[[Any], Any]) -> None:
    """
    Register a custom serializer for a specific type globally.

    This function registers a serializer that will be used by `safe_serialize` and
    other serialization functions when encountering objects of the specified type.

    Args:
        obj_type: The type to register a serializer for
        serializer_func: Function that converts the type to a serializable format

    Example:
        >>> from datetime import datetime
        >>> def serialize_datetime(dt: datetime) -> str:
        ...     return dt.strftime("%Y-%m-%d %H:%M:%S")
        >>> register_custom_serializer(datetime, serialize_datetime)
    """
    with _registry_lock:
        _custom_serializers[obj_type] = serializer_func


def register_custom_deserializer(
    obj_type: Type[Any], deserializer_func: Callable[[Any], Any]
) -> None:
    """
    Register a custom deserializer for a specific type globally.

    This function registers a deserializer that will be used by reconstruction functions
    when encountering serialized data that should be converted back to the original type.

    Args:
        obj_type: The type to register a deserializer for
        deserializer_func: Function that converts serialized data back to the original type

    Example:
        >>> from datetime import datetime
        >>> def deserialize_datetime(data: str) -> datetime:
        ...     return datetime.fromisoformat(data)
        >>> register_custom_deserializer(datetime, deserialize_datetime)
    """
    with _registry_lock:
        _custom_deserializers[obj_type] = deserializer_func


def lookup_custom_serializer(value: Any) -> Optional[Callable[[Any], Any]]:
    """
    Look up a registered serializer for a value's type.

    Args:
        value: The value to find a serializer for

    Returns:
        The registered serializer function, or None if not found

    Example:
        >>> serializer = lookup_custom_serializer(some_value)
        >>> if serializer:
        ...     result = serializer(some_value)
    """
    with _registry_lock:
        # Check exact type first
        if type(value) in _custom_serializers:
            return _custom_serializers[type(value)]

        # Check for base classes
        for base_type, serializer in _custom_serializers.items():
            if isinstance(value, base_type):
                return serializer

        return None


def lookup_custom_deserializer(obj_type: Type[Any]) -> Optional[Callable[[Any], Any]]:
    """
    Look up a registered deserializer for a type.

    Args:
        obj_type: The type to find a deserializer for

    Returns:
        The registered deserializer function, or None if not found

    Example:
        >>> deserializer = lookup_custom_deserializer(MyCustomType)
        >>> if deserializer:
        ...     result = deserializer(serialized_data)
    """
    with _registry_lock:
        # Check exact type first
        if obj_type in _custom_deserializers:
            return _custom_deserializers[obj_type]

        # Check for base classes - only if obj_type is actually a class
        if isinstance(obj_type, type):
            for base_type, deserializer in _custom_deserializers.items():
                if issubclass(obj_type, base_type):
                    return deserializer

    return None


def create_serializer_for_type(
    obj_type: Type[Any], serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """
    Create a serializer function that handles a specific type.

    Args:
        obj_type: The type to create a serializer for
        serializer_func: Function that serializes the type

    Returns:
        A serializer function that handles the specific type
    """

    def serializer(obj: Any) -> Any:
        if isinstance(obj, obj_type):
            return serializer_func(obj)
        return obj

    return serializer


def create_field_serializer(
    field_name: str, serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """
    Create a field_serializer method for a specific field.

    Args:
        field_name: Name of the field to serialize
        serializer_func: Function that serializes the field value

    Returns:
        A serializer function that can be used within field_serializer methods
    """

    def field_serializer_method(value: Any) -> Any:
        return serializer_func(value)

    return field_serializer_method


def serializable_field(serializer_func: Callable[[Any], Any]) -> Callable[[T], T]:
    """
    Decorator to mark a field as serializable with a custom serializer.

    DEPRECATED: This function is deprecated due to fundamental design issues
    with Pydantic v2. Use register_custom_serializer or manual field_serializer instead.

    Args:
        serializer_func: Function to serialize the field

    Returns:
        Decorator function (deprecated)

    Example:
        # DEPRECATED - Use register_custom_serializer instead
        >>> class MyModel(BaseModel):
        ...     @serializable_field(lambda x: x.to_dict())
        ...     complex_object: ComplexType
    """

    def decorator(field: T) -> T:
        # This is a no-op in the new system, but we keep it for backward compatibility
        return field

    return decorator


def _serialize_for_key(
    obj: Any,
    _seen: Optional[Set[int]] = None,
    default_serializer: Optional[Callable[[Any], Any]] = None,
) -> str:
    """
    Serialize an object for use as a dictionary key.

    This function is used internally by safe_serialize when serializing
    dictionary keys, ensuring they are always strings for JSON compatibility.

    Args:
        obj: The object to serialize for use as a key
        _seen: Internal set for circular reference detection (do not use directly)
        default_serializer: Optional custom serializer for unknown types

    Returns:
        A string representation of the object suitable for JSON object keys
    """
    # For keys, we need to ensure the result is a string for JSON compatibility
    serialized = safe_serialize(obj, default_serializer=default_serializer, _seen=_seen)

    # Convert to string to guarantee JSON-compatible keys
    return str(serialized)


def safe_deserialize(
    serialized_data: Any,
    target_type: Optional[Type[Any]] = None,
    default_deserializer: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """
    Safely deserialize an object with intelligent fallback handling.

    This function provides robust deserialization for:
    - Pydantic models (v1 and v2)
    - Dataclasses
    - Lists, tuples, sets, frozensets, dicts
    - Enums
    - Special float values (inf, -inf, nan)
    - Primitives (str, int, bool, None)
    - Datetime objects (datetime, date, time)
    - Bytes and memoryview objects
    - Complex numbers
    - Custom types registered via register_custom_deserializer

    Args:
        serialized_data: The serialized data to deserialize
        target_type: Optional type hint for the expected result type
        default_deserializer: Optional custom deserializer for unknown types

    Returns:
        The deserialized object

    Raises:
        TypeError: If object cannot be deserialized and no default_deserializer is provided

    Note:
        - Special float values (inf, -inf, nan) are converted from strings
        - Datetime objects are converted from ISO format strings
        - Bytes are converted from base64 strings
        - Complex numbers are converted from dict with 'real' and 'imag' keys
        - Custom types registered via register_custom_deserializer are automatically handled
    """
    if serialized_data is None:
        return None

    # Handle primitives
    if isinstance(serialized_data, (str, int, bool)):
        return serialized_data

    # Handle special float values
    if isinstance(serialized_data, str):
        if serialized_data == "nan":
            return float("nan")
        if serialized_data == "inf":
            return float("inf")
        if serialized_data == "-inf":
            return float("-inf")

    # Handle float
    if isinstance(serialized_data, float):
        return serialized_data

    # Handle lists
    if isinstance(serialized_data, list):
        return [safe_deserialize(item, None, default_deserializer) for item in serialized_data]

    # Handle dictionaries
    if isinstance(serialized_data, dict):
        # Check if this looks like a serialized custom type
        if target_type is not None:
            custom_deserializer = lookup_custom_deserializer(target_type)
            if custom_deserializer:
                try:
                    return custom_deserializer(serialized_data)
                except Exception:
                    pass  # Fall back to dict reconstruction

        # Reconstruct as dict
        return {
            safe_deserialize(k, None, default_deserializer): safe_deserialize(
                v, None, default_deserializer
            )
            for k, v in serialized_data.items()
        }

    # Handle datetime objects (from ISO format strings)
    if isinstance(serialized_data, str):
        try:
            from datetime import datetime

            # Try to parse as datetime
            dt = datetime.fromisoformat(serialized_data.replace("Z", "+00:00"))
            return dt
        except (ValueError, TypeError):
            pass

    # Handle complex numbers (from dict with 'real' and 'imag' keys)
    if (
        isinstance(serialized_data, dict)
        and "real" in serialized_data
        and "imag" in serialized_data
    ):
        try:
            return complex(serialized_data["real"], serialized_data["imag"])
        except (ValueError, TypeError):
            pass

    # Handle bytes (from base64 strings)
    if isinstance(serialized_data, str):
        try:
            import base64

            # Try to decode as base64
            decoded = base64.b64decode(serialized_data)
            return decoded
        except Exception:
            pass

    # Handle enums
    if target_type is not None and hasattr(target_type, "__members__"):
        # This looks like an enum
        try:
            return target_type(serialized_data)
        except (ValueError, TypeError):
            pass

    # Handle Pydantic models
    if target_type is not None and hasattr(target_type, "model_validate"):
        try:
            return target_type.model_validate(serialized_data)
        except Exception:
            pass

    # Handle dataclasses
    if target_type is not None and dataclasses.is_dataclass(target_type):
        try:
            return target_type(**serialized_data)
        except Exception:
            pass

    # Try default deserializer if provided
    if default_deserializer is not None:
        try:
            return default_deserializer(serialized_data)
        except Exception:
            pass

    # If we can't deserialize, return the original data
    return serialized_data


def safe_serialize(
    obj: Any,
    default_serializer: Optional[Callable[[Any], Any]] = None,
    _seen: Optional[Set[int]] = None,
) -> Any:
    """
    Safely serialize an object with intelligent fallback handling.

    This function provides robust serialization for:
    - Pydantic models (v1 and v2)
    - Dataclasses
    - Lists, tuples, sets, frozensets, dicts
    - Enums
    - Special float values (inf, -inf, nan)
    - Circular references
    - Primitives (str, int, bool, None)
    - Datetime objects (datetime, date, time)
    - Bytes and memoryview objects
    - Complex numbers
    - Functions and callables
    - Custom types registered via register_custom_serializer

    Args:
        obj: The object to serialize
        default_serializer: Optional custom serializer for unknown types
        _seen: Internal set for circular reference detection (do not use directly)

    Returns:
        JSON-serializable representation of the object

    Raises:
        TypeError: If object cannot be serialized and no default_serializer is provided

    Note:
        - Circular references are serialized as None
        - Roundtrip is not guaranteed for objects with circular/self-referential structures
        - Special float values (inf, -inf, nan) are converted to strings
        - Datetime objects are converted to ISO format strings
        - Bytes are converted to base64 strings
        - Complex numbers are converted to dict with 'real' and 'imag' keys
        - Functions are converted to their name or repr
        - Custom types registered via register_custom_serializer are automatically handled
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        # Circular reference detected - serialize as None
        return None
    _seen.add(obj_id)

    try:
        # Give priority to custom serializers over built-in handling
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer:
            return safe_serialize(custom_serializer(obj), default_serializer, _seen)

        # Handle None
        if obj is None:
            return None

        # Handle primitives
        if isinstance(obj, (str, int, bool)):
            return obj

        # Handle special float values
        if isinstance(obj, float):
            if math.isnan(obj):
                return "nan"
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj

        # Check for custom serializers in the global registry FIRST
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer:
            return safe_serialize(custom_serializer(obj), default_serializer, _seen)

        # Handle datetime objects
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()

        # Handle bytes and memoryview
        if isinstance(obj, (bytes, memoryview)):
            if isinstance(obj, memoryview):
                obj = obj.tobytes()
            import base64

            return base64.b64encode(obj).decode("ascii")

        # Handle complex numbers
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        # Handle functions and callables
        if callable(obj):
            if hasattr(obj, "__name__"):
                return obj.__name__
            else:
                return repr(obj)

        # Handle dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                k: safe_serialize(v, default_serializer, _seen)
                for k, v in dataclasses.asdict(obj).items()
            }

        # Handle enums
        if isinstance(obj, Enum):
            return obj.value

        # Handle Pydantic v2 models
        if hasattr(obj, "model_dump"):
            return safe_serialize(obj.model_dump(), default_serializer, _seen)

        # Handle Pydantic v1 models
        if HAS_PYDANTIC and isinstance(obj, BaseModel):
            return safe_serialize(obj.dict(), default_serializer, _seen)

        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                str(_serialize_for_key(k, _seen, default_serializer)): safe_serialize(
                    v, default_serializer, _seen
                )
                for k, v in obj.items()
            }

        # Handle sequences (list, tuple, set, frozenset)
        if isinstance(obj, (list, tuple)):
            return [safe_serialize(item, default_serializer, _seen) for item in obj]
        # Handle sets and frozensets with sorted order for deterministic output
        if isinstance(obj, (set, frozenset)):
            return [
                safe_serialize(item, default_serializer, _seen) for item in sorted(obj, key=str)
            ]

        # Handle custom serializer if provided
        if default_serializer:
            return default_serializer(obj)

        # If we get here, the type is not supported
        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable. "
            f"Consider providing a custom default_serializer or registering a custom serializer "
            f"using register_custom_serializer."
        )

    finally:
        # Always remove from seen set to allow reuse
        _seen.discard(obj_id)


def robust_serialize(obj: Any) -> Any:
    """
    Robust serialization that handles all common Python types.

    This is a convenience wrapper around safe_serialize that provides
    a more permissive fallback for unknown types.

    Args:
        obj: The object to serialize

    Returns:
        JSON-serializable representation of the object
    """

    def fallback_serializer(obj: Any) -> Any:
        """Fallback serializer that converts unknown types to dict if Pydantic, else string representation."""
        # Extra robust: if this is a Pydantic model, use model_dump or dict
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if HAS_PYDANTIC and isinstance(obj, BaseModel):
            return obj.dict()
        return f"<unserializable: {type(obj).__name__}>"

    return safe_serialize(obj, default_serializer=fallback_serializer)


def serialize_to_json(obj: Any, **kwargs: Any) -> str:
    """
    Serialize an object to a JSON string.

    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string representation of the object

    Raises:
        TypeError: If the object cannot be serialized to JSON
    """
    serialized = safe_serialize(obj)
    return json.dumps(serialized, sort_keys=True, **kwargs)


def serialize_to_json_robust(obj: Any, **kwargs: Any) -> str:
    """
    Serialize an object to a JSON string with robust fallback handling.

    This version will never fail, but may serialize unknown types as strings.

    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string representation of the object
    """
    serialized = robust_serialize(obj)
    return json.dumps(serialized, **kwargs)


def reset_custom_serializer_registry() -> None:
    """Reset the global custom serializer and deserializer registries (for testing only)."""
    with _registry_lock:
        _custom_serializers.clear()
        _custom_deserializers.clear()
