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
    from pydantic import BaseModel as PydanticBaseModel

    HAS_PYDANTIC = True
except (ImportError, ModuleNotFoundError):
    # Define a lightweight placeholder class to satisfy isinstance checks without assigning to a type
    HAS_PYDANTIC = False

    class PydanticBaseModel:  # type: ignore[no-redef]
        pass


from flujo.type_definitions.common import JSONObject


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
    _recursion_depth: int = 0,
    mode: str = "default",
) -> str:
    """
    Serialize an object for use as a dictionary key.

    This function is used internally by safe_serialize when serializing
    dictionary keys, ensuring they are always strings for JSON compatibility.
    Handles circular references robustly and always uses custom serializers for keys.
    """
    PRIMITIVE_TYPES = (str, int, float, bool, type(None))
    if _seen is None:
        _seen = set()
    if isinstance(obj, PRIMITIVE_TYPES):
        return str(obj)

    # Handle tuples specially for keys to preserve format
    if isinstance(obj, tuple):
        try:
            # Convert tuple to string directly
            items = []
            for item in obj:
                if isinstance(item, PRIMITIVE_TYPES):
                    items.append(str(item))
                else:
                    items.append(
                        _serialize_for_key(
                            item, _seen, default_serializer, _recursion_depth + 1, mode
                        )
                    )
            return f"({', '.join(items)})"
        except Exception:
            return str(obj)

    obj_id = id(obj)
    custom_serializer = lookup_custom_serializer(obj)
    added_to_seen = False  # Track if we added obj_id to _seen in this call
    try:
        # Always use the custom serializer for keys, even if circular
        if custom_serializer:
            serialized = custom_serializer(obj)
            # If the custom serializer returns a non-primitive, serialize it with increased recursion depth
            if not isinstance(serialized, PRIMITIVE_TYPES):
                serialized = safe_serialize(
                    serialized,
                    default_serializer=default_serializer,
                    _seen=_seen,
                    _recursion_depth=_recursion_depth + 1,
                    mode=mode,
                )
            return str(serialized)
        # If no custom serializer, check for circularity
        if obj_id in _seen:
            return "<circular-key>"
        _seen.add(obj_id)
        added_to_seen = True
        serialized = safe_serialize(
            obj,
            default_serializer=default_serializer,
            _seen=_seen,
            _recursion_depth=_recursion_depth + 1,
            mode=mode,
        )
        return str(serialized)
    except Exception:
        return "<circular-key>"
    finally:
        if not custom_serializer and added_to_seen:
            _seen.discard(obj_id)


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
    - Datetime objects (datetime, date, time) when target_type is not explicitly a datetime type
    - Bytes and memoryview objects (base64-decoded when target_type requests bytes-like)
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

    # Handle primitives (non-str)
    if isinstance(serialized_data, (bool, int, float)):
        return serialized_data

    # Handle strings with special cases (nan/inf, datetime, base64)
    if isinstance(serialized_data, str):
        if target_type is not None:
            from datetime import date, datetime, time

            if target_type in {datetime, date, time}:
                return serialized_data
        if serialized_data == "nan":
            return float("nan")
        if serialized_data == "inf":
            return float("inf")
        if serialized_data == "-inf":
            return float("-inf")
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(serialized_data.replace("Z", "+00:00"))
            return dt
        except (ValueError, TypeError):
            pass
        if target_type in {bytes, bytearray, memoryview}:
            try:
                import base64
                import binascii

                decoded = base64.b64decode(serialized_data, validate=True)
                if target_type is bytearray:
                    return bytearray(decoded)
                if target_type is memoryview:
                    return memoryview(decoded)
                return decoded
            except (ValueError, binascii.Error):
                pass
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
        except Exception as e:
            try:
                from flujo.infra import telemetry as _telemetry

                _telemetry.logfire.warning(
                    f"safe_deserialize: validation failed for {getattr(target_type, '__name__', target_type)} with error {type(e).__name__}: {e}. Data type: {type(serialized_data).__name__}"
                )
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
    _recursion_depth: int = 0,
    circular_ref_placeholder: Any = "<circular-ref>",
    mode: str = "default",
) -> Any:
    """
    Safely serialize an object with intelligent fallback handling.

    This is the unified serialization function that handles all edge cases and special types,
    consolidating logic from BaseModel.model_dump and other specialized serializers.

    Handles circular references robustly with mode-specific behavior:
    - "default" mode: Returns appropriate placeholders (None for objects, {} for dicts, [] for lists)
    - "python" mode: Mirrors "default" for circulars to align with BaseModel.model_dump
    - "cache" mode: Returns "<ClassName circular>" placeholders
    - Custom modes: Uses the circular_ref_placeholder parameter

    Features:
    - Circular reference detection with mode-specific handling
    - Custom serializer registry support
    - Comprehensive type handling (datetime, Enum, complex, bytes, etc.)
    - Pydantic model support with manual field serialization for Flujo models
    - Dataclass support
    - Collections (list, tuple, dict, set) with recursive serialization
    - Callable object handling with mock detection
    - Error recovery with fallback strategies

    Args:
        obj: The object to serialize
        default_serializer: Optional fallback serializer for unknown types
        _seen: Internal circular reference tracking set
        _recursion_depth: Internal recursion depth counter
        circular_ref_placeholder: What to return for circular references (overridden by mode)
        mode: Serialization mode ("default", "cache", or custom)

    Returns:
        Serialized representation of the object

    Raises:
        TypeError: For unserializable objects when no fallback is available
    """
    PRIMITIVE_TYPES = (str, int, float, bool, type(None))
    if _seen is None:
        _seen = set()

    # Handle datetime objects with mode-aware behavior
    if isinstance(obj, (datetime, date, time)):
        # Preserve native types in python mode; cast to ISO string otherwise
        return obj if mode == "python" else obj.isoformat()

    # Handle Enum objects specifically - don't add to _seen
    if isinstance(obj, Enum):
        try:
            return obj.value
        except (AttributeError, TypeError):
            return str(obj)

    # Preserve certain standard library types verbatim in python mode for type fidelity
    try:
        import uuid as _uuid

        if isinstance(obj, _uuid.UUID):
            return obj if mode == "python" else str(obj)
    except Exception:
        pass
    # Preserve Decimal in python mode to maintain numeric fidelity; otherwise stringify for general serialization
    try:
        from decimal import Decimal as _Decimal

        if isinstance(obj, _Decimal):
            return obj if mode == "python" else str(obj)
    except Exception:
        pass

    # Determine if this is a Flujo BaseModel (used in multiple branches)
    is_flujo_model = False
    if HAS_PYDANTIC:
        try:
            from flujo.domain.base_model import BaseModel as FlujoBaseModel

            is_flujo_model = isinstance(obj, FlujoBaseModel)
        except ImportError:
            is_flujo_model = False

    # Handle circular references for non-primitive types
    if not isinstance(obj, PRIMITIVE_TYPES):
        obj_id = id(obj)
        if obj_id in _seen:
            # If a specific circular_ref_placeholder was provided (not the default), use it
            if circular_ref_placeholder != "<circular-ref>":
                return circular_ref_placeholder

            # Handle mode-specific circular reference behavior for default placeholder
            if mode == "cache":
                # Generate class-specific circular reference marker
                class_name = getattr(obj.__class__, "__name__", type(obj).__name__)
                return f"<{class_name} circular>"
            elif mode in {"default", "python"}:
                if is_flujo_model:
                    return None
                elif isinstance(obj, dict):
                    return {}
                elif isinstance(obj, (list, tuple)):
                    return []
                elif isinstance(obj, (set, frozenset)):
                    return []
                else:
                    # For other objects in default/python modes, use the circular_ref_placeholder
                    return circular_ref_placeholder
            else:
                # For other modes, use the provided placeholder
                return circular_ref_placeholder
        _seen.add(obj_id)

    # Limit recursion depth to prevent stack overflow
    if _recursion_depth > 50:
        return f"<max-depth-exceeded: {type(obj).__name__}>"

    try:
        # Check for custom serializers first
        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer:
            try:
                serialized_result = custom_serializer(obj)
                return safe_serialize(
                    serialized_result,
                    default_serializer,
                    _seen,
                    _recursion_depth + 1,
                    circular_ref_placeholder,
                    mode,
                )
            except Exception as e:
                # Re-raise specific exceptions that should not be caught
                if isinstance(e, (ValueError, TypeError)) and "failed" in str(e).lower():
                    raise
                # For other exceptions, fall through to default handling
                pass

        # Prefer Pydantic model_dump for BaseModel instances; then serialize recursively
        if HAS_PYDANTIC and isinstance(obj, PydanticBaseModel) and not is_flujo_model:
            # Avoid recursive model_dump (may trigger circular errors); serialize __dict__ using python mode
            try:
                raw = getattr(obj, "__dict__", obj)
                return safe_serialize(
                    raw,
                    default_serializer,
                    _seen,
                    _recursion_depth + 1,
                    circular_ref_placeholder,
                    "python",
                )
            except Exception:
                pass

        if obj is None:
            return None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj):
                return "nan"
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj
        if isinstance(obj, (bytes, memoryview)):
            if isinstance(obj, memoryview):
                obj = obj.tobytes()
            import base64

            return base64.b64encode(obj).decode("ascii")
        # Preserve Decimal in python mode to maintain numeric fidelity
        try:
            from decimal import Decimal as _Decimal

            if isinstance(obj, _Decimal):
                return obj if mode == "python" else float(obj)
        except Exception:
            pass
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        # Handle mock objects specifically before anything else
        # Be more specific about Mock detection to avoid false positives with test classes
        if hasattr(obj, "__class__") and (
            obj.__class__.__name__
            in ("Mock", "MagicMock", "AsyncMock", "NonCallableMock", "CallableMixin")
            or (
                hasattr(obj.__class__, "__module__")
                and obj.__class__.__module__
                and "unittest.mock" in obj.__class__.__module__
            )
        ):
            # Handle mock objects for testing with improved detection
            try:
                # print(f"DEBUG: Found mock object early: {obj.__class__.__name__}")
                mock_result = serialize_mock_object(obj, mode, _seen)
                # print(f"DEBUG: Mock serialization succeeded early: {type(mock_result)}")
                return mock_result
            except Exception:
                # If mock serialization fails, fall back to string representation
                # print(f"DEBUG: Mock serialization failed early: {e}")
                return f"Mock({obj.__class__.__name__})"

        if callable(obj):
            if hasattr(obj, "__name__"):
                return obj.__name__
            else:
                return repr(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                k: safe_serialize(
                    v,
                    default_serializer,
                    _seen,
                    _recursion_depth + 1,
                    circular_ref_placeholder,
                    mode,
                )
                for k, v in dataclasses.asdict(obj).items()
            }
        # Handle Pydantic models with enhanced Flujo BaseModel support
        if hasattr(obj, "model_dump") or (HAS_PYDANTIC and isinstance(obj, PydanticBaseModel)):
            if is_flujo_model:
                # This is a Flujo BaseModel - build a dict manually from declared fields
                # to avoid triggering Pydantic's strict serializer (which emits warnings
                # for intentional runtime type widenings used by Flujo).
                try:
                    # Build a dict manually from declared fields to avoid Pydantic's strict validation
                    base_dict = {}
                    for name in getattr(obj.__class__, "model_fields", {}):
                        base_dict[name] = getattr(obj, name, None)
                    result = {}
                    for name, value in base_dict.items():
                        try:
                            result[name] = safe_serialize(
                                value,
                                default_serializer,
                                _seen,
                                _recursion_depth + 1,
                                circular_ref_placeholder,
                                mode,
                            )
                        except TypeError as e:
                            if "not serializable" in str(e):
                                result[name] = f"<unserializable: {type(value).__name__}>"
                            else:
                                raise
                        except Exception:
                            result[name] = f"<unserializable: {type(value).__name__}>"
                    return result
                except Exception:
                    return str(obj)
            else:
                # For other Pydantic models, use their native model_dump and process carefully
                try:
                    # Prefer python-mode dump to preserve Python types (datetime, UUID, Decimal)
                    if hasattr(obj, "model_dump"):
                        try:
                            model_dict = obj.model_dump(mode="python")
                        except TypeError:
                            # Older signatures without mode parameter
                            model_dict = obj.model_dump()
                    else:
                        if hasattr(obj, "dict"):
                            model_dict = obj.dict()
                        else:
                            model_dict = {}
                    # For each value in the model dict, serialize recursively with per-field fallback
                    result = {}
                    for k, v in model_dict.items():
                        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
                            # Basic types - keep as-is
                            result[k] = v
                        else:
                            try:
                                result[k] = safe_serialize(
                                    v,
                                    default_serializer,
                                    _seen,
                                    _recursion_depth + 1,
                                    circular_ref_placeholder,
                                    "python",
                                )
                            except TypeError as e:
                                if "not serializable" in str(e):
                                    result[k] = f"<unserializable: {type(v).__name__}>"
                                else:
                                    raise
                            except Exception:
                                result[k] = f"<unserializable: {type(v).__name__}>"
                    return result
                except Exception:
                    return str(obj)
        # Handle AgentResponse objects with proper field extraction
        if hasattr(obj, "output") and hasattr(obj, "usage"):
            # This looks like an AgentResponse object
            return serialize_agent_response(obj, mode)
        # Handle objects with cost_usd and token_counts (like UsageResponse)
        if hasattr(obj, "cost_usd") and hasattr(obj, "token_counts"):
            return {
                "cost_usd": getattr(obj, "cost_usd", 0.0),
                "token_counts": getattr(obj, "token_counts", 0),
                "output": getattr(obj, "output", None),
            }
        # Handle dictionaries
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                try:
                    key_str = str(
                        _serialize_for_key(k, _seen, default_serializer, _recursion_depth + 1, mode)
                    )
                    result[key_str] = safe_serialize(
                        v,
                        default_serializer,
                        _seen,
                        _recursion_depth + 1,
                        circular_ref_placeholder,
                        mode,
                    )
                except TypeError as e:
                    if "not serializable" in str(e) and _recursion_depth > 0:
                        # Only convert to string if we're in a nested context
                        # At the top level, let the TypeError propagate
                        result[str(k)] = f"<unserializable: {type(v).__name__}>"
                    else:
                        raise
            return result
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [
                safe_serialize(
                    item,
                    default_serializer,
                    _seen,
                    _recursion_depth + 1,
                    circular_ref_placeholder,
                    mode,
                )
                for item in obj
            ]
        # Handle sets
        if isinstance(obj, (set, frozenset)):
            return [
                safe_serialize(
                    item,
                    default_serializer,
                    _seen,
                    _recursion_depth + 1,
                    circular_ref_placeholder,
                    mode,
                )
                for item in obj
            ]

        # Handle regular objects with __dict__ attributes
        # Only serialize as dict for specific known types, not arbitrary objects
        if hasattr(obj, "__dict__") and not isinstance(obj, type):
            # Check if there's a custom serializer first
            if lookup_custom_serializer(obj) is not None:
                # Let the custom serializer handle it - should have been processed above
                pass
            elif hasattr(obj, "model_dump") or (
                HAS_PYDANTIC and isinstance(obj, PydanticBaseModel)
            ):
                # Let Pydantic models be handled by the model_dump path above
                pass
            else:
                # For other objects with __dict__, raise TypeError to enforce explicit serialization
                raise TypeError(
                    f"Object of type {type(obj).__name__} is not serializable. "
                    f"Register a custom serializer using register_custom_serializer({type(obj).__name__}, lambda obj: obj.__dict__) or provide a default_serializer."
                )
        # If we get here, we have an unknown type
        if default_serializer:
            return default_serializer(obj)
        else:
            # For objects with __dict__ that aren't handled by custom serializers,
            # we should raise TypeError as expected by tests
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                raise TypeError(
                    f"Object of type {type(obj).__name__} is not serializable. "
                    f"Register a custom serializer using register_custom_serializer({type(obj).__name__}, lambda obj: obj.__dict__) or provide a default_serializer."
                )
            else:
                return handle_unknown_type(obj)
    except Exception as e:
        # Enhanced error handling with better context
        if default_serializer:
            try:
                return default_serializer(obj)
            except Exception:
                return f"<serialization-error: {type(obj).__name__} - {str(e)}>"
        else:
            # Re-raise specific exceptions that should be propagated
            if isinstance(e, TypeError):
                raise
            elif isinstance(e, ValueError) and "failed" in str(e).lower():
                raise
            else:
                return f"<serialization-error: {type(obj).__name__} - {str(e)}>"
    finally:
        # Clean up the seen set only at the top level to ensure proper circular reference tracking
        if _recursion_depth == 0 and _seen:
            _seen.clear()
        # For non-primitives, remove from seen set when exiting this scope to allow reprocessing
        # in different branches of the object graph
        elif not isinstance(obj, PRIMITIVE_TYPES) and obj is not None:
            try:
                _seen.discard(id(obj))
            except (TypeError, AttributeError):
                # Some objects might not have stable IDs or might be unhashable
                pass


def serialize_agent_response(response: Any, mode: str = "default") -> JSONObject:
    """
    Serialize AgentResponse objects properly with proper field extraction.

    Args:
        response: An object that looks like an AgentResponse (has output/content and usage attributes)

    Returns:
        A serializable dictionary representation of the AgentResponse
    """
    result: JSONObject = {
        "content": getattr(response, "content", getattr(response, "output", None)),
        "metadata": {},
    }
    metadata: JSONObject = result["metadata"]

    # Handle usage information if present
    if hasattr(response, "usage"):
        if callable(response.usage):
            try:
                usage_info = response.usage()
                if hasattr(usage_info, "request_tokens") and hasattr(usage_info, "response_tokens"):
                    metadata["usage"] = {
                        "request_tokens": usage_info.request_tokens,
                        "response_tokens": usage_info.response_tokens,
                    }
                elif hasattr(usage_info, "model_dump"):
                    # Handle Pydantic usage models
                    metadata["usage"] = usage_info.model_dump()
                else:
                    # Fallback for other usage objects
                    metadata["usage"] = safe_serialize(usage_info, mode=mode)
            except Exception:
                # If usage() fails, just skip it
                pass
        else:
            # Direct usage attribute
            metadata["usage"] = safe_serialize(response.usage, mode=mode)

    # Handle additional attributes that might be present
    for attr in [
        "_prompt_tokens",
        "_completion_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "token_counts",
    ]:
        if hasattr(response, attr):
            metadata[attr] = getattr(response, attr)

    # Handle metadata field if present
    if hasattr(response, "metadata"):
        if hasattr(response.metadata, "model_dump"):
            metadata.update(response.metadata.model_dump())
        else:
            metadata.update(safe_serialize(response.metadata, mode=mode))

    return result


def serialize_mock_object(
    mock_obj: Any, mode: str = "default", _seen: Optional[Set[int]] = None
) -> JSONObject:
    """
    Serialize Mock objects for testing scenarios with improved detection.

    Args:
        mock_obj: A Mock object to serialize
        mode: Serialization mode
        _seen: Set of seen object IDs for circular reference detection

    Returns:
        A serializable dictionary representation of the Mock object
    """
    result: JSONObject = {
        "type": "Mock",
        "class_name": type(mock_obj).__name__,
        "module": getattr(mock_obj, "__module__", "unknown"),
        "attributes": {},
    }

    # Only serialize user-added attributes, not internal mock attributes
    if hasattr(mock_obj, "__dict__"):
        for key, value in mock_obj.__dict__.items():
            # Only include non-mock attributes (user-added ones)
            if not key.startswith("_mock_") and not key.startswith("method_calls"):
                try:
                    # Simple serialization for basic types only
                    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
                        result["attributes"][key] = value
                    else:
                        result["attributes"][key] = str(value)
                except Exception:
                    result["attributes"][key] = f"<unserializable: {type(value).__name__}>"

    # Add basic mock state info
    try:
        if hasattr(mock_obj, "_mock_name") and mock_obj._mock_name:
            result["attributes"]["_mock_name"] = str(mock_obj._mock_name)
        if hasattr(mock_obj, "_mock_called"):
            result["attributes"]["_mock_called"] = bool(mock_obj._mock_called)
        if hasattr(mock_obj, "_mock_call_count"):
            result["attributes"]["_mock_call_count"] = int(mock_obj._mock_call_count)
    except Exception:
        # If we can't get mock state, that's fine
        pass

    return result


def handle_unknown_type(obj: Any) -> str:
    """
    Handle unknown types with helpful error messages and custom serializer suggestions.

    Args:
        obj: The object that couldn't be serialized

    Returns:
        A helpful error message string
    """
    obj_type = type(obj).__name__
    obj_module = getattr(obj, "__module__", "unknown")

    # Provide specific suggestions based on the object type
    if hasattr(obj, "__dict__"):
        suggestion = f"Consider registering a custom serializer: register_custom_serializer({obj_type}, lambda obj: obj.__dict__)"
    elif hasattr(obj, "model_dump"):
        suggestion = "Object has model_dump method - this should be handled automatically"
    elif hasattr(obj, "__class__") and "Mock" in obj.__class__.__name__:
        suggestion = "Mock object detected - this should be handled automatically"
    elif hasattr(obj, "__slots__"):
        suggestion = f"Object uses __slots__ - consider: register_custom_serializer({obj_type}, lambda obj: {{name: getattr(obj, name, None) for name in obj.__slots__}})"
    elif hasattr(obj, "__getstate__"):
        suggestion = f"Object has __getstate__ method - consider: register_custom_serializer({obj_type}, lambda obj: obj.__getstate__())"
    else:
        suggestion = f"Consider registering a custom serializer: register_custom_serializer({obj_type}, your_serializer_function)"

    return f"<unserializable: {obj_type} from {obj_module}> - {suggestion}"


def robust_serialize(obj: Any, circular_ref_placeholder: Any = "<circular-ref>") -> Any:
    """
    Robust serialization for logging/debugging only. Never use for production data.

    Rules:
    - Preserve primitives (str, int, float, bool, None) exactly
    - Preserve built-in containers (dict, list, tuple, set, frozenset) as structured data
    - For other objects (custom classes), return a string representation of the serialized form
    - On error, return a concise fallback string
    """
    PRIMITIVES = (str, int, float, bool, type(None))
    BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)
    try:
        # If the original object is a primitive, return as-is
        if isinstance(obj, PRIMITIVES):
            return obj

        # Pydantic/Flujo models: preserve structured data (avoid unsafe hasattr)
        try:
            md = getattr(obj, "model_dump")
        except Exception:
            md = None
        if md is not None:
            return safe_serialize(obj, circular_ref_placeholder=circular_ref_placeholder)

        # If a custom serializer is registered for this type, use it first to avoid
        # triggering attribute access (e.g., __getattr__ raising) in safe_serialize.
        from .serialization import lookup_custom_serializer as _lookup

        _ser = _lookup(obj)
        if _ser is not None:
            try:
                interim = _ser(obj)
                # If the custom serializer already returns a primitive/container, use it directly
                if isinstance(interim, (str, int, float, bool, type(None))) or isinstance(
                    interim, (dict, list, tuple, set, frozenset)
                ):
                    return interim
                # Otherwise, serialize the serializer output
                return safe_serialize(interim, circular_ref_placeholder=circular_ref_placeholder)
            except Exception:
                # Fall through to generic paths
                pass

        # Built-in containers: preserve structured data
        if isinstance(obj, BUILTIN_CONTAINERS):
            return safe_serialize(obj, circular_ref_placeholder=circular_ref_placeholder)

        # Custom/unknown types: prefer structured result when available; else concise string
        serialized: Any = safe_serialize(
            obj,
            circular_ref_placeholder=circular_ref_placeholder,
        )
        if isinstance(serialized, PRIMITIVES) or isinstance(serialized, BUILTIN_CONTAINERS):
            return serialized
        return repr(serialized)
    except Exception:
        return f"<unserializable: {type(obj).__name__}>"


def serialize_to_json(obj: Any, mode: str = "default", **kwargs: Any) -> str:
    """
    Serialize an object to a JSON string.

    Args:
        obj: The object to serialize
        mode: Serialization mode ("default" or "cache")
        **kwargs: Additional arguments to pass to json.dumps

    Returns:
        JSON string representation of the object

    Raises:
        TypeError: If the object cannot be serialized to JSON
    """
    serialized: Any = safe_serialize(obj, mode=mode)
    return json.dumps(serialized, sort_keys=True, **kwargs)


def serialize_to_json_robust(obj: Any, **kwargs: Any) -> str:
    """
    Serialize an object to a JSON string using robust_serialize.
    Ensures the output is always valid JSON for roundtrip.
    """
    import json

    return json.dumps(robust_serialize(obj), **kwargs)


def reset_custom_serializer_registry() -> None:
    """Reset the global custom serializer and deserializer registries (for testing only)."""
    with _registry_lock:
        _custom_serializers.clear()
        _custom_deserializers.clear()
