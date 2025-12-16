"""Serialization utilities for Flujo (Pydantic-v2 native)."""

from __future__ import annotations

import dataclasses
import json
import math
import threading
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Set, TypeVar
from uuid import UUID

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel as PydanticBaseModel

    HAS_PYDANTIC = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - compatibility
    HAS_PYDANTIC = False

    class PydanticBaseModel:  # type: ignore[no-redef]
        pass


T = TypeVar("T")

# ---------------------------------------------------------------------------
# Custom serializer/deserializer registry
# ---------------------------------------------------------------------------
_custom_serializers: Dict[type[Any], Callable[[Any], Any]] = {}
_custom_deserializers: Dict[type[Any], Callable[[Any], Any]] = {}
_registry_lock = threading.Lock()


def register_custom_serializer(obj_type: type[Any], serializer_func: Callable[[Any], Any]) -> None:
    """Register a serializer for a specific type."""

    with _registry_lock:
        _custom_serializers[obj_type] = serializer_func


def register_custom_deserializer(
    obj_type: type[Any], deserializer_func: Callable[[Any], Any]
) -> None:
    """Register a deserializer for a specific type."""

    with _registry_lock:
        _custom_deserializers[obj_type] = deserializer_func


def lookup_custom_serializer(value: Any) -> Optional[Callable[[Any], Any]]:
    """Return the first matching serializer for ``value``'s type (exact or subclass)."""

    with _registry_lock:
        exact = _custom_serializers.get(type(value))
        if exact is not None:
            return exact
        for base, serializer in _custom_serializers.items():
            if isinstance(value, base):
                return serializer
        return None


def lookup_custom_deserializer(obj_type: type[Any]) -> Optional[Callable[[Any], Any]]:
    """Return the first matching deserializer for ``obj_type`` (exact or subclass)."""

    with _registry_lock:
        exact = _custom_deserializers.get(obj_type)
        if exact is not None:
            return exact
        for base, deserializer in _custom_deserializers.items():
            try:
                if issubclass(obj_type, base):
                    return deserializer
            except Exception:
                continue
    return None


def reset_custom_serializer_registry() -> None:
    """Reset serializer/deserializer registries (testing helper)."""

    with _registry_lock:
        _custom_serializers.clear()
        _custom_deserializers.clear()


def create_serializer_for_type(
    obj_type: type[Any], serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Return a serializer that dispatches to ``serializer_func`` when type matches."""

    def serializer(obj: Any) -> Any:
        if isinstance(obj, obj_type):
            return serializer_func(obj)
        return obj

    return serializer


def create_field_serializer(
    field_name: str, serializer_func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Helper for Pydantic ``@field_serializer`` style usage."""

    def field_serializer_method(value: Any) -> Any:  # pragma: no cover - simple passthrough
        return serializer_func(value)

    return field_serializer_method


def serializable_field(serializer_func: Callable[[Any], Any]) -> Callable[[T], T]:
    """Deprecated decorator retained for backward compatibility."""

    def decorator(field: T) -> T:  # pragma: no cover - compatibility shim
        return field

    return decorator


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def safe_deserialize(
    serialized_data: Any,
    target_type: Optional[type[Any]] = None,
    default_deserializer: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Best-effort deserialization counterpart for ``_serialize_for_json``."""

    if serialized_data is None:
        return None

    if isinstance(serialized_data, (bool, int, float)):
        return serialized_data

    if isinstance(serialized_data, str):
        if target_type is float:
            if serialized_data == "nan":
                return float("nan")
            if serialized_data == "inf":
                return float("inf")
            if serialized_data == "-inf":
                return float("-inf")
            return serialized_data

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
                return serialized_data

        return serialized_data

    if isinstance(serialized_data, list):
        return [safe_deserialize(item, None, default_deserializer) for item in serialized_data]

    if isinstance(serialized_data, dict):
        if "real" in serialized_data and "imag" in serialized_data and len(serialized_data) == 2:
            try:
                return complex(serialized_data["real"], serialized_data["imag"])
            except (ValueError, TypeError):
                pass

        if target_type is not None:
            custom_deserializer = lookup_custom_deserializer(target_type)
            if custom_deserializer:
                try:
                    return custom_deserializer(serialized_data)
                except (TypeError, ValueError):
                    pass

        return {
            safe_deserialize(k, None, default_deserializer): safe_deserialize(
                v, None, default_deserializer
            )
            for k, v in serialized_data.items()
        }

    if target_type is not None and hasattr(target_type, "__members__"):
        try:
            return target_type(serialized_data)
        except (ValueError, TypeError):
            pass

    if target_type is not None and hasattr(target_type, "model_validate"):
        try:
            return target_type.model_validate(serialized_data)
        except Exception as exc:  # pragma: no cover - defensive log path
            try:
                from flujo.infra import telemetry as _telemetry

                _telemetry.logfire.warning(
                    "safe_deserialize: validation failed for %s: %s",
                    getattr(target_type, "__name__", target_type),
                    exc,
                )
            except Exception:
                pass

    if target_type is not None and dataclasses.is_dataclass(target_type):
        try:
            return target_type(**serialized_data)
        except Exception:
            pass

    if default_deserializer is not None:
        try:
            return default_deserializer(serialized_data)
        except Exception:
            pass

    return serialized_data


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
JsonValue = str | int | float | bool | None | Dict[str, Any] | list[Any]


def _normalize_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _json_serialize_impl(
    obj: Any,
    *,
    mode: str = "json",
    default_serializer: Optional[Callable[[Any], Any]] = None,
    circular_ref_placeholder: str | None = "<circular-ref>",
    strict: bool = False,
    bytes_mode: Literal["base64", "utf8"] = "base64",
    allow_object_dict: bool = False,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> JsonValue:
    """Convert arbitrary objects into JSON-safe structures.

    Internal implementation for flujo. External users should use
    ``model_dump(mode="json")`` for Pydantic models.

    - Pydantic models use ``model_dump`` with ``mode="json"`` by default.
    - Dataclasses are converted via ``asdict``.
    - Enums, UUID, Decimal, datetime/date/time are normalized to string (unless ``mode="python"``).
    - Circular references return "<circular-ref>".
    """

    if _seen is None:
        _seen = set()
    if _depth > 64:
        return f"<max-depth-exceeded:{type(obj).__name__}>"

    def _is_domain_model(value: Any) -> bool:
        try:
            from flujo.domain.base_model import BaseModel as DomainBaseModel

            return isinstance(value, DomainBaseModel)
        except Exception:
            return False

    def _cycle_value(value: Any) -> JsonValue:
        if _is_domain_model(value):
            if mode == "cache" or strict:
                return f"<{type(value).__name__} circular>"
            return None
        if mode == "cache":
            return f"<{type(value).__name__} circular>"
        if circular_ref_placeholder is None:
            return None
        if HAS_PYDANTIC and isinstance(value, PydanticBaseModel):
            return "<circular>"
        return circular_ref_placeholder

    custom = lookup_custom_serializer(obj)
    if custom:
        obj_id = id(obj)
        if obj_id in _seen:
            return _cycle_value(obj)
        try:
            custom_result = custom(obj)
        except Exception:
            if strict:
                raise
            custom_result = None
        else:
            if custom_result is not obj:
                _seen.add(obj_id)
                try:
                    return _json_serialize_impl(
                        custom_result,
                        mode=mode,
                        default_serializer=default_serializer,
                        circular_ref_placeholder=circular_ref_placeholder,
                        strict=strict,
                        bytes_mode=bytes_mode,
                        allow_object_dict=allow_object_dict,
                        _seen=_seen,
                        _depth=_depth + 1,
                    )
                finally:
                    _seen.discard(obj_id)

    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _normalize_float(obj)

    obj_id = id(obj)
    if obj_id in _seen:
        return _cycle_value(obj)

    _seen.add(obj_id)
    try:
        if isinstance(obj, Enum):
            return _json_serialize_impl(
                obj.value,
                mode=mode,
                default_serializer=default_serializer,
                circular_ref_placeholder=circular_ref_placeholder,
                strict=strict,
                bytes_mode=bytes_mode,
                allow_object_dict=allow_object_dict,
                _seen=_seen,
                _depth=_depth + 1,
            )

        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()

        if isinstance(obj, (UUID, Decimal)):
            return str(obj)

        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (bytes, bytearray, memoryview)):
            if bytes_mode == "utf8":
                try:
                    return bytes(obj).decode("utf-8")
                except Exception:
                    return bytes(obj).decode("utf-8", errors="replace")
            import base64

            return base64.b64encode(bytes(obj)).decode("utf-8")

        if HAS_PYDANTIC and isinstance(obj, PydanticBaseModel):
            self_refs: set[str] = set()
            try:
                for field_name in getattr(obj, "__pydantic_fields__", {}):
                    try:
                        if getattr(obj, field_name) is obj:
                            self_refs.add(field_name)
                    except Exception:
                        continue
            except Exception:
                self_refs = set()
            try:
                dumped = PydanticBaseModel.model_dump(
                    obj, mode="json" if mode != "python" else "python"
                )
                for ref in self_refs:
                    dumped[ref] = _cycle_value(obj)
                return _json_serialize_impl(
                    dumped,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            except Exception:
                fallback_data = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
                if not fallback_data:
                    return f"<unserializable: {type(obj).__name__}>"
                return _json_serialize_impl(
                    fallback_data,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )

        if dataclasses.is_dataclass(obj):
            if isinstance(obj, type):
                raise TypeError(
                    f"Cannot serialize dataclass type {obj.__name__}; provide an instance instead."
                )
            return _json_serialize_impl(
                dataclasses.asdict(obj),
                mode=mode,
                default_serializer=default_serializer,
                circular_ref_placeholder=circular_ref_placeholder,
                strict=strict,
                bytes_mode=bytes_mode,
                allow_object_dict=allow_object_dict,
                _seen=_seen,
                _depth=_depth + 1,
            )

        if isinstance(obj, Mapping):
            out: Dict[str, Any] = {}
            for key, value in obj.items():
                key_str = str(
                    _json_serialize_impl(
                        key,
                        mode=mode,
                        default_serializer=default_serializer,
                        circular_ref_placeholder=circular_ref_placeholder,
                        strict=strict,
                        bytes_mode=bytes_mode,
                        allow_object_dict=allow_object_dict,
                        _seen=_seen,
                        _depth=_depth + 1,
                    )
                )
                out[key_str] = _json_serialize_impl(
                    value,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            return out

        if isinstance(obj, (set, frozenset)):
            return [
                _json_serialize_impl(
                    item,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in obj
            ]

        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray, memoryview)):
            return [
                _json_serialize_impl(
                    item,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in obj
            ]

        if callable(obj):
            if hasattr(obj, "__name__"):
                return str(obj.__name__)
            if hasattr(obj, "__qualname__"):
                return str(obj.__qualname__)
            return repr(obj)

        if default_serializer is not None:
            try:
                return _json_serialize_impl(
                    default_serializer(obj),
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            except Exception:
                pass

        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump()
                return _json_serialize_impl(
                    dumped,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            if strict:
                raise TypeError(
                    f"{type(obj).__name__} is not serializable; "
                    "register_custom_serializer or provide default_serializer"
                )
            data = {
                k: v for k, v in vars(obj).items() if not (k.startswith("__") and k.endswith("__"))
            }
            if _depth == 0 and allow_object_dict and data:
                return _json_serialize_impl(
                    data,
                    mode=mode,
                    default_serializer=default_serializer,
                    circular_ref_placeholder=circular_ref_placeholder,
                    strict=strict,
                    bytes_mode=bytes_mode,
                    allow_object_dict=allow_object_dict,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            if _depth == 0:
                if data and ("output" in data or "content" in data):
                    return _json_serialize_impl(
                        data,
                        mode=mode,
                        default_serializer=default_serializer,
                        circular_ref_placeholder=circular_ref_placeholder,
                        strict=strict,
                        bytes_mode=bytes_mode,
                        allow_object_dict=allow_object_dict,
                        _seen=_seen,
                        _depth=_depth + 1,
                    )
                if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
                    try:
                        return str(obj.to_dict())
                    except Exception:
                        return str(obj)
                raise TypeError(
                    f"{type(obj).__name__} is not serializable; "
                    "register_custom_serializer or provide default_serializer"
                )
            if data:
                return str(obj)
            return f"<unserializable: {type(obj).__name__}>"

        if _depth > 0:
            if strict:
                raise TypeError(
                    f"{type(obj).__name__} is not serializable; "
                    "register_custom_serializer or provide default_serializer"
                )
            return f"<unserializable: {type(obj).__name__}>"
        raise TypeError(
            f"{type(obj).__name__} is not serializable; "
            "register_custom_serializer or provide default_serializer"
        )
    finally:
        _seen.discard(obj_id)


# ---------------------------------------------------------------------------
# Internal-use functions (primary serialization API for flujo internals)
# External users should use model_dump(mode="json") directly.
# ---------------------------------------------------------------------------


def _serialize_for_json(
    obj: Any,
    *,
    mode: str = "json",
    default_serializer: Optional[Callable[[Any], Any]] = None,
    circular_ref_placeholder: str | None = "<circular-ref>",
    strict: bool = False,
    bytes_mode: Literal["base64", "utf8"] = "base64",
    allow_object_dict: bool = False,
    _seen: Optional[Set[int]] = None,
) -> JsonValue:
    """Internal serialization for flujo modules.

    External code should use model_dump(mode="json") for Pydantic models.
    """
    return _json_serialize_impl(
        obj,
        mode=mode,
        default_serializer=default_serializer,
        circular_ref_placeholder=circular_ref_placeholder,
        strict=strict,
        bytes_mode=bytes_mode,
        allow_object_dict=allow_object_dict,
        _seen=_seen,
    )


def _robust_serialize_internal(
    obj: Any,
    *,
    circular_ref_placeholder: str | None = "<circular-ref>",
    bytes_mode: Literal["base64", "utf8"] = "base64",
    allow_object_dict: bool = False,
) -> JsonValue | str:
    """Internal robust serializer that never raises.

    For flujo internals that need never-raise serialization.
    """
    try:
        return _json_serialize_impl(
            obj,
            circular_ref_placeholder=circular_ref_placeholder,
            bytes_mode=bytes_mode,
            allow_object_dict=allow_object_dict,
        )
    except Exception:  # noqa: BLE001 - intentional blanket catch; serializer must never raise
        return f"<unserializable: {type(obj).__name__}>"


def _serialize_to_json_internal(obj: Any, *, mode: str = "json", **kwargs: Any) -> str:
    """Internal JSON string serializer for flujo modules.

    For flujo internals that need JSON string output.
    """
    return json.dumps(_json_serialize_impl(obj, mode=mode), sort_keys=True, **kwargs)


__all__ = [
    # Serialization utilities (public API)
    "create_field_serializer",
    "create_serializer_for_type",
    "lookup_custom_serializer",
    "lookup_custom_deserializer",
    "register_custom_serializer",
    "register_custom_deserializer",
    "reset_custom_serializer_registry",
    "safe_deserialize",
    "serializable_field",
    # Internal functions (for flujo modules only)
    "_serialize_for_json",
    "_robust_serialize_internal",
    "_serialize_to_json_internal",
]
