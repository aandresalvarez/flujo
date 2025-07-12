"""Serialization utilities for Flujo."""

import dataclasses
import json
import math
from datetime import datetime, date, time
from enum import Enum
from typing import Any, Callable, Optional, Set


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
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        # Circular reference detected - serialize as None
        return None
    _seen.add(obj_id)

    try:
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
        if hasattr(obj, "dict"):
            return safe_serialize(obj.dict(), default_serializer, _seen)

        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                safe_serialize(k, default_serializer, _seen): safe_serialize(
                    v, default_serializer, _seen
                )
                for k, v in obj.items()
            }

        # Handle sequences (list, tuple, set, frozenset)
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [safe_serialize(item, default_serializer, _seen) for item in obj]

        # Handle custom serializer if provided
        if default_serializer:
            return default_serializer(obj)

        # If we get here, the type is not supported
        raise TypeError(
            f"Object of type {type(obj).__name__} is not serializable. "
            f"Consider providing a custom default_serializer."
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
        if hasattr(obj, "dict"):
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
    return json.dumps(serialized, **kwargs)


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


def serializable_field(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "serializable_field is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )


def create_serializer_for_type(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "create_serializer_for_type is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )


def register_custom_serializer(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "register_custom_serializer is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )


def lookup_custom_serializer(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "lookup_custom_serializer is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )


def create_field_serializer(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "create_field_serializer is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )


def _serialize_for_key(*args: Any, **kwargs: Any) -> None:
    """DEPRECATED: This function is no longer supported. Use robust serialization instead."""
    raise NotImplementedError(
        "_serialize_for_key is deprecated and no longer supported. "
        "Use robust serialization (safe_serialize) instead."
    )
