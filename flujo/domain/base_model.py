"""Custom BaseModel for all flujo domain models with intelligent fallback serialization."""

from typing import Any, Set, ClassVar, Optional
from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from datetime import datetime, date, time
from enum import Enum
from types import FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType


class BaseModel(PydanticBaseModel):
    """BaseModel for all flujo domain models with intelligent fallback serialization."""

    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    def _is_unknown_type(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, int, float, bool, list, dict, datetime, Enum)):
            return False
        return (
            callable(value)
            or isinstance(value, (complex, set, frozenset, bytes, memoryview))
            or (hasattr(value, "__dict__") and not hasattr(value, "model_dump"))
        )

    def model_dump(
        self, *, mode: str = "default", _seen: Optional[Set[int]] = None, **kwargs: Any
    ) -> Any:
        if _seen is None:
            _seen = set()
        obj_id = id(self)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{self.__class__.__name__} circular>"
            return None
        _seen.add(obj_id)
        try:
            result = {}
            for name, field in getattr(self.__class__, "model_fields", {}).items():
                value = getattr(self, name)
                # Handle circular references at the field level
                if value is None or isinstance(value, (str, int, float, bool)):
                    result[name] = value
                elif isinstance(value, (datetime, date, time)):
                    # Handle datetime objects specifically - don't add to _seen
                    result[name] = value.isoformat()
                elif isinstance(value, Enum):
                    # Handle Enum objects specifically - don't add to _seen
                    try:
                        result[name] = value.value
                    except (AttributeError, TypeError):
                        result[name] = str(value)
                elif isinstance(value, (list, tuple)):
                    # Handle lists and tuples with circular reference detection
                    if isinstance(value, list):
                        serialized_list = []
                        for item in value:
                            if item is None or isinstance(item, (str, int, float, bool)):
                                serialized_list.append(item)
                            elif isinstance(item, (datetime, date, time)):
                                serialized_list.append(item.isoformat())
                            elif isinstance(item, Enum):
                                try:
                                    serialized_list.append(item.value)
                                except (AttributeError, TypeError):
                                    serialized_list.append(str(item))
                            else:
                                item_id = id(item)
                                if item_id in _seen:
                                    if mode == "cache":
                                        serialized_list.append(f"<{type(item).__name__} circular>")
                                    else:
                                        serialized_list.append(None)
                                else:
                                    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
                                        if hasattr(item, "_safe_serialize_with_seen"):
                                            serialized_list.append(item.model_dump(mode=mode, _seen=_seen))
                                        else:
                                            serialized_list.append(item.model_dump(mode=mode))
                                    else:
                                        serialized_list.append(self._safe_serialize_with_seen(item, _seen, mode=mode))
                        result[name] = serialized_list
                    else:  # tuple
                        serialized_tuple = []
                        for item in value:
                            if item is None or isinstance(item, (str, int, float, bool)):
                                serialized_tuple.append(item)
                            elif isinstance(item, (datetime, date, time)):
                                serialized_tuple.append(item.isoformat())
                            elif isinstance(item, Enum):
                                try:
                                    serialized_tuple.append(item.value)
                                except (AttributeError, TypeError):
                                    serialized_tuple.append(str(item))
                            else:
                                item_id = id(item)
                                if item_id in _seen:
                                    if mode == "cache":
                                        serialized_tuple.append(f"<{type(item).__name__} circular>")
                                    else:
                                        serialized_tuple.append(None)
                                else:
                                    if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
                                        if hasattr(item, "_safe_serialize_with_seen"):
                                            serialized_tuple.append(item.model_dump(mode=mode, _seen=_seen))
                                        else:
                                            serialized_tuple.append(item.model_dump(mode=mode))
                                    else:
                                        serialized_tuple.append(self._safe_serialize_with_seen(item, _seen, mode=mode))
                        result[name] = tuple(serialized_tuple)
                else:
                    # Handle other types with circular reference detection
                    value_id = id(value)
                    if value_id in _seen:
                        if mode == "cache":
                            result[name] = f"<{type(value).__name__} circular>"
                        else:
                            result[name] = None
                    else:
                        from flujo.utils.serialization import lookup_custom_serializer
                        custom_serializer = lookup_custom_serializer(value)
                        if custom_serializer:
                            serialized = custom_serializer(value)
                            result[name] = self._safe_serialize_with_seen(serialized, _seen, mode=mode)
                        elif hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
                            if hasattr(value, "_safe_serialize_with_seen"):
                                result[name] = value.model_dump(mode=mode, _seen=_seen)
                            else:
                                result[name] = value.model_dump(mode=mode)
                        else:
                            result[name] = self._safe_serialize_with_seen(value, _seen, mode=mode)
            return result
        finally:
            _seen.discard(obj_id)

    def _safe_serialize_with_seen(self, obj: Any, _seen: Set[int], mode: str = "default") -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Handle datetime objects specifically to prevent infinite recursion - don't add to _seen
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        # Handle Enum objects - don't add to _seen
        if isinstance(obj, Enum):
            try:
                return obj.value
            except (AttributeError, TypeError):
                return str(obj)
        obj_id = id(obj)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{type(obj).__name__} circular>"
            return None
        _seen.add(obj_id)
        try:
            # Handle lists - check each item individually for circular references
            if isinstance(obj, list):
                result = []
                for item in obj:
                    if item is None or isinstance(item, (str, int, float, bool)):
                        result.append(item)
                    else:
                        item_id = id(item)
                        if item_id in _seen:
                            if mode == "cache":
                                result.append(f"<{type(item).__name__} circular>")
                            else:
                                result.append(None)
                        else:
                            result.append(self._safe_serialize_with_seen(item, _seen, mode=mode))
                return result
            # Handle tuples - check each item individually for circular references
            if isinstance(obj, tuple):
                result = []
                for item in obj:
                    if item is None or isinstance(item, (str, int, float, bool)):
                        result.append(item)
                    else:
                        item_id = id(item)
                        if item_id in _seen:
                            if mode == "cache":
                                result.append(f"<{type(item).__name__} circular>")
                            else:
                                result.append(None)
                        else:
                            result.append(self._safe_serialize_with_seen(item, _seen, mode=mode))
                return tuple(result)
            # Handle dictionaries - check each value individually for circular references
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Handle key
                    if k is None or isinstance(k, (str, int, float, bool)):
                        key = k
                    else:
                        key_id = id(k)
                        if key_id in _seen:
                            if mode == "cache":
                                key = f"<{type(k).__name__} circular>"
                            else:
                                key = None
                        else:
                            key = self._safe_serialize_with_seen(k, _seen, mode=mode)
                    # Handle value
                    if v is None or isinstance(v, (str, int, float, bool)):
                        value = v
                    else:
                        value_id = id(v)
                        if value_id in _seen:
                            if mode == "cache":
                                value = f"<{type(v).__name__} circular>"
                            else:
                                value = None
                        else:
                            value = self._safe_serialize_with_seen(v, _seen, mode=mode)
                    if key is not None:  # Only add if key is not None
                        result[key] = value
                return result
            # Handle objects with model_dump method
            if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
                if hasattr(obj, "_safe_serialize_with_seen"):
                    return obj.model_dump(mode=mode, _seen=_seen)
                else:
                    return obj.model_dump(mode=mode)
            # Check for custom serializers last, after handling known types
            from flujo.utils.serialization import lookup_custom_serializer
            custom_serializer = lookup_custom_serializer(obj)
            if custom_serializer:
                serialized = custom_serializer(obj)
                # If the custom serializer returns the same object, we have a problem
                if serialized is obj:
                    # Handle common cases where custom serializer returns the same object
                    if isinstance(obj, (datetime, date, time)):
                        return obj.isoformat()
                    elif isinstance(obj, Enum):
                        try:
                            return obj.value
                        except (AttributeError, TypeError):
                            return str(obj)
                    else:
                        # For other cases, try to convert to string representation
                        return str(obj)
                # Otherwise, recursively serialize the result
                return self._safe_serialize_with_seen(serialized, _seen, mode=mode)
            # Handle unknown types
            if self._is_unknown_type(obj):
                return self._serialize_single_unknown_type(obj, _seen, mode=mode)
            return obj
        finally:
            _seen.discard(obj_id)

    def model_dump_json(self, **kwargs: Any) -> str:
        data = self.model_dump(**kwargs)
        import json

        return json.dumps(data, **kwargs)

    def _process_serialized_data(
        self, data: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        return self._safe_serialize_with_seen(data, _seen or set(), mode=mode)

    def _recursively_serialize_dict(
        self, obj: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        return self._safe_serialize_with_seen(obj, _seen or set(), mode=mode)

    def _serialize_single_unknown_type(
        self, value: Any, _seen: Optional[Set[int]] = None, mode: str = "default"
    ) -> Any:
        if _seen is None:
            _seen = set()
        if value is None:
            return None
        from flujo.utils.serialization import lookup_custom_serializer

        custom_serializer = lookup_custom_serializer(value)
        if custom_serializer:
            serialized_result = custom_serializer(value)
            return self._recursively_serialize_dict(serialized_result, _seen, mode=mode)
        if isinstance(value, (str, int, float, bool)):
            return value
        obj_id = id(value)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{type(value).__name__} circular>"
            return None
        _seen.add(obj_id)
        try:
            if callable(value):
                if isinstance(
                    value,
                    (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType),
                ):
                    module = getattr(value, "__module__", "<unknown>")
                    qualname = getattr(value, "__qualname__", repr(value))
                    return f"{module}.{qualname}"
                else:
                    return repr(value)
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, complex):
                real_part = int(value.real) if value.real == int(value.real) else value.real
                imag_part = int(value.imag) if value.imag == int(value.imag) else value.imag
                real_str = str(real_part)
                imag_str = str(imag_part)
                if real_str.endswith(".0"):
                    real_str = real_str[:-2]
                if imag_str.endswith(".0"):
                    imag_str = imag_str[:-2]
                return f"{real_str}+{imag_str}j"
            if isinstance(value, (set, frozenset)):
                return list(value)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
        finally:
            _seen.discard(obj_id)
        return repr(value)
