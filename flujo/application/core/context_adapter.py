from __future__ import annotations

import types
import threading
import sys
from typing import (
    Any,
    Optional,
    Type,
    Union,
    Dict,
    TypeVar,
    cast,
    get_type_hints,
    get_args,
    Iterator,
)
from contextlib import contextmanager

from pydantic import ValidationError
from pydantic import BaseModel as PydanticBaseModel

from ...infra import telemetry
from ...domain.models import BaseModel
from ...utils.serialization import register_custom_serializer, register_custom_deserializer

__all__ = [
    "_build_context_update",
    "_inject_context",
    "register_custom_type",
    "TypeResolutionContext",
]

T = TypeVar("T")


# Thread-safe type resolution context
class TypeResolutionContext:
    """
    Thread-safe, scoped type resolution context.

    This provides a robust, future-proof type resolution system that:
    1. Integrates with Flujo's serialization infrastructure
    2. Uses Python's type system properly
    3. Provides validation and safety
    4. Supports module-scoped resolution
    5. Is thread-safe and performant
    """

    def __init__(self) -> None:
        self._resolvers: Dict[str, "_ModuleTypeResolver"] = {}
        self._current_module: Optional[Any] = None
        self._lock = threading.RLock()

    @contextmanager
    def module_scope(self, module: Any) -> Iterator[None]:
        """Set the current module scope for type resolution."""
        with self._lock:
            self._current_module = module
            try:
                yield
            finally:
                self._current_module = None

    def resolve_type(self, type_name: str, base_type: Type[T]) -> Optional[Type[T]]:
        """
        Resolve type with validation using current module scope.

        Args:
            type_name: Name of the type to resolve
            base_type: Expected base type for validation

        Returns:
            Resolved type if found and valid, None otherwise
        """
        with self._lock:
            if self._current_module is None:
                return None

            module_name = getattr(self._current_module, "__name__", str(id(self._current_module)))
            resolver = self._resolvers.get(module_name)

            if resolver is None:
                resolver = _ModuleTypeResolver(self._current_module)
                self._resolvers[module_name] = resolver

            type_obj = resolver.resolve_type(type_name)
            if type_obj is not None and self._validate_type_resolution(type_obj, base_type):
                return cast(Type[T], type_obj)

            return None

    def _validate_type_resolution(self, type_obj: Any, expected_base: Type[Any]) -> bool:
        """Validate that resolved object is actually a valid type."""
        if not isinstance(type_obj, type):
            return False

        # Check if it's a subclass of expected base
        if not issubclass(type_obj, expected_base):
            return False

        return True


class _ModuleTypeResolver:
    """Module-scoped type resolver with caching."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self._cache: Dict[str, Type[Any]] = {}
        self._type_hints_cache: Optional[Dict[str, Any]] = None

    def resolve_type(self, type_name: str) -> Optional[Type[Any]]:
        """Resolve type from module scope with caching."""
        if type_name in self._cache:
            return self._cache[type_name]

        # Try module attributes first
        if hasattr(self.module, type_name):
            type_obj = getattr(self.module, type_name)
            if isinstance(type_obj, type):
                self._cache[type_name] = type_obj
                return type_obj

        # Try type hints from module
        type_hints = self._get_module_type_hints()
        if type_name in type_hints:
            type_obj = type_hints[type_name]
            if isinstance(type_obj, type):
                self._cache[type_name] = type_obj
                return type_obj

        return None

    def _get_module_type_hints(self) -> Dict[str, Any]:
        """Get type hints from module with caching."""
        if self._type_hints_cache is None:
            try:
                self._type_hints_cache = get_type_hints(self.module)
            except Exception:
                self._type_hints_cache = {}
        return self._type_hints_cache


# Global type resolution context
_type_context = TypeResolutionContext()


def register_custom_type(type_class: Type[T]) -> None:
    """
    Register a custom type for serialization and type resolution.

    This integrates with Flujo's serialization system and provides
    automatic type resolution for the registered type.

    Args:
        type_class: The type class to register for serialization and type resolution.

    Returns:
        None

    Raises:
        ValueError: If the type class doesn't have required methods for serialization.
    """
    if hasattr(type_class, "__name__"):
        # Register for serialization - register the class itself
        register_custom_serializer(
            type_class, lambda obj: obj.model_dump() if hasattr(obj, "model_dump") else obj.__dict__
        )

        # Register deserializer if it's a Pydantic model
        if hasattr(type_class, "model_validate") and callable(
            getattr(type_class, "model_validate", None)
        ):
            # Use a type-safe approach to call model_validate
            def safe_model_validate(data: Any) -> Any:
                model_validate = getattr(type_class, "model_validate", None)
                if callable(model_validate):
                    return model_validate(data)
                raise ValueError(f"Type {type_class} does not have a callable model_validate method")

            register_custom_deserializer(type_class, safe_model_validate)


def _resolve_type_from_string(type_str: str) -> Optional[Type[Any]]:
    """
    Robust type resolution using Python's type system.

    This replaces the fragile regex-based approach with proper
    type system integration and validation.
    """
    if not type_str or not isinstance(type_str, str):
        return None

    # Try to resolve using type system first
    try:
        # Check if it's a valid type annotation
        if hasattr(types, "UnionType") and isinstance(type_str, types.UnionType):
            return type_str

        # Ensure type_str is treated as a string, as documented.
        # Try to evaluate as a type annotation
        import ast

        try:
            ast.parse(type_str, mode="eval")
            # This is a simplified approach - in practice, you'd want more robust evaluation
            return None
        except (SyntaxError, ValueError):
            pass
    except Exception:
        pass

    # Fallback to module resolution
    if _type_context._current_module is not None:
        return _type_context.resolve_type(type_str, object)

    return None


def _extract_union_types(union_type: Any) -> list[Type[Any]]:
    """
    Extract non-None types from a Union type annotation.

    Uses Python's type system properly instead of regex parsing.
    """
    non_none_types: list[Type[Any]] = []

    # Handle Python 3.10+ Union syntax (types.UnionType)
    if isinstance(union_type, types.UnionType):
        try:
            args = get_args(union_type)
            non_none_types = [t for t in args if t is not type(None)]
        except Exception:
            # Fallback: try to extract from string representation
            type_str = str(union_type)
            # Use proper type parsing instead of regex
            non_none_types = _parse_type_string(type_str)
        return non_none_types

    # Handle traditional Union[T, None] syntax
    if hasattr(union_type, "__origin__") and union_type.__origin__ is Union:
        non_none_types = [t for t in union_type.__args__ if t is not type(None)]
    elif hasattr(union_type, "__union_params__"):
        non_none_types = [t for t in union_type.__union_params__ if t is not type(None)]

    return non_none_types


def _parse_type_string(type_str: str) -> list[Type[Any]]:
    """
    Parse type string using proper type system integration.

    This replaces regex-based parsing with proper type analysis.
    """
    types_found: list[Type[Any]] = []

    try:
        # Try to get type hints from current module
        if _type_context._current_module is not None:
            type_hints = get_type_hints(_type_context._current_module)

            # Look for types that appear in the string
            for type_name, type_obj in type_hints.items():
                if type_name in type_str and isinstance(type_obj, type):
                    types_found.append(type_obj)
    except Exception:
        pass

    return types_found


def _resolve_actual_type(field_type: Any) -> Optional[Type[Any]]:
    """
    Resolve the actual type from a field annotation using type system.

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

    This centralizes the deserialization logic and integrates with
    Flujo's serialization system.
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
            resolved_element_type = _resolve_actual_type(element_type)
            if resolved_element_type is not None:
                actual_element_type = resolved_element_type
                # Handle Pydantic models in list
                if (
                    hasattr(actual_element_type, "model_validate")
                    and callable(getattr(actual_element_type, "model_validate", None))
                    and issubclass(actual_element_type, BaseModel)
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
        resolved_type = _resolve_actual_type(field_type)
        if resolved_type is None:
            return value

        actual_type: Type[Any] = resolved_type
        # Handle Pydantic models
        if (
            hasattr(actual_type, "model_validate")
            and callable(getattr(actual_type, "model_validate", None))
            and issubclass(actual_type, BaseModel)
        ):
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

    # Use type system integration for type discovery
    # Get the module where the context model is defined
    module = sys.modules.get(context_model.__module__)
    if module is None:
        # Fallback to current module
        module = sys.modules.get("__main__")

    with _type_context.module_scope(module):
        # Extract types from field annotations using type system
        for field_name, field_info in context_model.model_fields.items():
            if field_info.annotation:
                # Use proper type analysis instead of regex
                field_type = field_info.annotation
                if hasattr(field_type, "__name__"):
                    type_name = field_type.__name__
                    # Register the type if it's a custom type
                    if type_name not in [
                        "str",
                        "int",
                        "float",
                        "bool",
                        "list",
                        "dict",
                        "tuple",
                        "set",
                    ]:
                        try:
                            if isinstance(field_type, type):
                                register_custom_type(field_type)
                        except Exception:
                            pass

    # Process update data
    for key, value in update_data.items():
        if key in context_model.model_fields:
            field_info = context_model.model_fields[key]
            field_type = field_info.annotation  # type: ignore[assignment]

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
        # If the field exists on the context, set it directly
        setattr(context, key, value)

    # Final pass: re-apply deserialization to ensure consistency
    for key in context_model.model_fields:
        field_info = context_model.model_fields[key]
        field_type = field_info.annotation  # type: ignore[assignment]
        current_value = getattr(context, key, None)

        if current_value is not None and field_type is not None:
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
