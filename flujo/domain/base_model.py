"""Custom BaseModel for all flujo domain models with unified serialization."""

from typing import Any, ClassVar
from pydantic import BaseModel as PydanticBaseModel, ConfigDict


class BaseModel(PydanticBaseModel):
    """BaseModel for all flujo domain models with unified serialization.

    This model delegates all serialization logic to flujo.utils.serialization
    to maintain a single source of truth for serialization behavior.

    Note: Flujo domain models intentionally carry runtime objects (agents, hooks,
    validators, resources) that are not Pydantic models. Keeping
    `arbitrary_types_allowed=True` at this boundary enables ergonomic composition
    while serialization remains centrally controlled via `model_dump()`.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, *, mode: str = "default", **kwargs: Any) -> Any:
        """
        Unified model serialization using native Pydantic serialization, then normalized
        via the shared JSON-safe serializer (supports custom serializer registry).
        """
        try:
            data = super().model_dump(mode=mode, **kwargs)
        except ValueError as exc:
            # Gracefully handle circular references by falling back to python mode
            if "Circular reference detected" not in str(exc):
                raise
            try:
                data = super().model_dump(mode="python", serialize_as_any=True, **kwargs)
            except Exception:
                data = self.__dict__
        try:
            from flujo.state.backends.base import _serialize_for_json

            normalized = _serialize_for_json(data, _seen={id(self)})

            def _strip_placeholders(value: Any) -> Any:
                if (
                    isinstance(value, str)
                    and value.startswith("<")
                    and value.endswith(" circular>")
                ):
                    return None
                if isinstance(value, list):
                    return [_strip_placeholders(v) for v in value]
                if isinstance(value, tuple):
                    return tuple(_strip_placeholders(v) for v in value)
                if isinstance(value, dict):
                    return {k: _strip_placeholders(v) for k, v in value.items()}
                return value

            if mode != "cache":
                return _strip_placeholders(normalized)
            return normalized
        except Exception:
            return data

    def model_dump_json(self, **kwargs: Any) -> str:
        """
        Serialize model to JSON string using unified serialization.

        Args:
            **kwargs: Arguments passed to json.dumps

        Returns:
            JSON string representation of the model
        """
        from flujo.utils.serialization import _serialize_to_json_internal

        # Extract mode if present in kwargs, default to "default"
        mode = kwargs.pop("mode", "default")
        data = self.model_dump(mode=mode, **kwargs)
        return _serialize_to_json_internal(data, mode=mode, **kwargs)
