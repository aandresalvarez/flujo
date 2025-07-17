"""Custom BaseModel for all flujo domain models with intelligent fallback serialization."""

from typing import Any, Set, ClassVar, Optional
from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from datetime import datetime
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
                from flujo.utils.serialization import lookup_custom_serializer

                custom_serializer = lookup_custom_serializer(value)
                if custom_serializer:
                    serialized = custom_serializer(value)
                    result[name] = self._safe_serialize_with_seen(serialized, _seen, mode=mode)
                    continue
                if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
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
        obj_id = id(obj)
        if obj_id in _seen:
            if mode == "cache":
                return f"<{type(obj).__name__} circular>"
            return None
        from flujo.utils.serialization import lookup_custom_serializer

        custom_serializer = lookup_custom_serializer(obj)
        if custom_serializer:
            serialized = custom_serializer(obj)
            return self._safe_serialize_with_seen(serialized, _seen, mode=mode)
        if isinstance(obj, list):
            return [self._safe_serialize_with_seen(item, _seen, mode=mode) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._safe_serialize_with_seen(item, _seen, mode=mode) for item in obj)
        if isinstance(obj, dict):
            _seen.add(obj_id)
            try:
                return {
                    self._safe_serialize_with_seen(
                        k, _seen, mode=mode
                    ): self._safe_serialize_with_seen(v, _seen, mode=mode)
                    for k, v in obj.items()
                }
            finally:
                _seen.discard(obj_id)
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            if hasattr(obj, "_safe_serialize_with_seen"):
                return obj.model_dump(mode=mode, _seen=_seen)
            else:
                return obj.model_dump(mode=mode)
        if self._is_unknown_type(obj):
            return self._serialize_single_unknown_type(obj, _seen, mode=mode)
        return obj

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
                    value, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)
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
