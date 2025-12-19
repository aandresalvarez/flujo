from __future__ import annotations
from flujo.type_definitions.common import JSONObject

import threading
import types
from contextlib import contextmanager
from typing import (
    Iterator,
    Optional,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)
from collections.abc import MutableMapping

from pydantic import ValidationError
from pydantic import BaseModel as PydanticBaseModel

from ....infra import telemetry
from ....domain.models import BaseModel, ExecutedCommandLog, ImportArtifacts
from ....utils.serialization import register_custom_serializer, register_custom_deserializer
from ....utils.scratchpad import SCRATCHPAD_REMOVED_MESSAGE, update_contains_scratchpad

__all__ = [
    "TypeResolutionContext",
    "_build_context_update",
    "_inject_context",
    "register_custom_type",
]

T = TypeVar("T")


def _is_import_artifacts(obj: object) -> TypeGuard[ImportArtifacts]:
    return isinstance(obj, ImportArtifacts)


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
        self._resolvers: dict[str, _ModuleTypeResolver] = {}
        self._current_module: object | None = None
        self._lock = threading.RLock()
        self._global_type_cache: dict[str, type[object]] = {}

    @contextmanager
    def module_scope(self, module: object) -> Iterator[None]:
        """Set the current module scope for type resolution."""
        with self._lock:
            self._current_module = module
            try:
                yield
            finally:
                self._current_module = None

    def resolve_type(self, type_name: str, base_type: type[T]) -> type[T] | None:
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

            # Check global cache first for frequently resolved types
            cache_key = f"{type_name}:{base_type.__name__}"
            if cache_key in self._global_type_cache:
                cached_type = self._global_type_cache[cache_key]
                if self._validate_type_resolution(cached_type, base_type):
                    return cached_type

            module_name = getattr(self._current_module, "__name__", str(id(self._current_module)))
            resolver = self._resolvers.get(module_name)

            if resolver is None:
                resolver = _ModuleTypeResolver(self._current_module)
                self._resolvers[module_name] = resolver

            type_obj = resolver.resolve_type(type_name)
            if type_obj is not None and self._validate_type_resolution(type_obj, base_type):
                # Add to global cache for future lookups
                self._global_type_cache[cache_key] = type_obj
                return type_obj

            return None

    def _validate_type_resolution(
        self, type_obj: object, expected_base: type[T]
    ) -> TypeGuard[type[T]]:
        """Validate that resolved object is actually a valid type."""
        if not isinstance(type_obj, type):
            return False
        return issubclass(type_obj, expected_base)

    def clear_global_cache(self) -> None:
        """Clear the global type cache."""
        with self._lock:
            self._global_type_cache.clear()


class _ModuleTypeResolver:
    """Module-scoped type resolver with caching."""

    def __init__(self, module: object) -> None:
        self.module = module
        self._cache: dict[str, type[object]] = {}
        self._type_hints_cache: Optional[JSONObject] = None

    def resolve_type(self, type_name: str) -> type[object] | None:
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

    def _get_module_type_hints(self) -> JSONObject:
        """Get type hints from module with caching."""
        if self._type_hints_cache is None:
            try:
                self._type_hints_cache = get_type_hints(self.module)
            except Exception:
                self._type_hints_cache = {}
        return self._type_hints_cache


# Global type resolution context
_type_context = TypeResolutionContext()


def register_custom_type(type_class: type[T]) -> None:
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
        # Check if this is a Flujo BaseModel to avoid circular dependency
        from flujo.domain.base_model import BaseModel as FlujoBaseModel

        def serialize_custom_type(obj: object) -> object:
            """Serializer that avoids circular dependency with Flujo BaseModel."""
            if isinstance(obj, FlujoBaseModel):
                try:
                    return obj.model_dump(mode="json")
                except Exception:
                    pass
                try:
                    result = {}
                    for field_name in getattr(obj.__class__, "model_fields", {}):
                        result[field_name] = getattr(obj, field_name, None)
                    return result
                except Exception:
                    return getattr(obj, "__dict__", obj)
            if hasattr(obj, "model_dump"):
                try:
                    return obj.model_dump(mode="json")
                except Exception:
                    pass
            return getattr(obj, "__dict__", obj)

        # Register for serialization
        register_custom_serializer(type_class, serialize_custom_type)

        # Register deserializer if it's a Pydantic model
        if hasattr(type_class, "model_validate") and callable(
            getattr(type_class, "model_validate", None)
        ):
            # Use a type-safe approach to call model_validate
            def safe_model_validate(data: object) -> object:
                model_validate = getattr(type_class, "model_validate", None)
                if callable(model_validate):
                    return model_validate(data)
                raise ValueError(
                    f"Type {type_class} does not have a callable model_validate method"
                )

            register_custom_deserializer(type_class, safe_model_validate)


def _resolve_type_from_string(type_str: str) -> type[object] | None:
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


def _extract_union_types(union_type: object) -> list[object]:
    """
    Extract non-None types from a Union type annotation.

    Uses Python's type system properly instead of regex parsing.
    """
    non_none_types: list[object] = []

    # Handle Python 3.10+ Union syntax (types.UnionType)
    if isinstance(union_type, types.UnionType):
        try:
            args = get_args(union_type)
            non_none_types = [t for t in args if t is not type(None)]
        except Exception:
            # Fallback: try to extract from string representation
            type_str = str(union_type)
            # Use proper type parsing instead of regex
            non_none_types = [t for t in _parse_type_string(type_str)]
        return non_none_types

    # Handle traditional Union[T, None] syntax
    origin = getattr(union_type, "__origin__", None)
    if origin is Union:
        args_obj = getattr(union_type, "__args__", ())
        if isinstance(args_obj, tuple):
            non_none_types = [t for t in args_obj if t is not type(None)]
    else:
        params_obj = getattr(union_type, "__union_params__", ())
        if isinstance(params_obj, tuple):
            non_none_types = [t for t in params_obj if t is not type(None)]

    return non_none_types


def _parse_type_string(type_str: str) -> list[type[object]]:
    """
    Parse type string using proper type system integration.

    This replaces regex-based parsing with proper type analysis.
    """
    types_found: list[type[object]] = []

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


def _resolve_actual_type(field_type: object) -> type[object] | None:
    """
    Resolve the actual type from a field annotation using type system.

    This is a robust replacement for the fragile type resolution logic.
    """
    if field_type is None:
        return None

    # Handle string forward references
    if isinstance(field_type, str):
        if field_type == "ExecutedCommandLog":
            return ExecutedCommandLog
        resolved = _resolve_type_from_string(field_type)
        if resolved is not None:
            return resolved

    # Handle Union types (compatible with Python 3.9+)
    # - typing.Union[...] uses __origin__ == typing.Union
    # - PEP604 unions (A | B) are instances of types.UnionType
    if isinstance(field_type, types.UnionType):
        non_none_types = [t for t in get_args(field_type) if t is not type(None)]
        if non_none_types:
            first = non_none_types[0]
            if isinstance(first, str):
                if first == "ExecutedCommandLog":
                    return ExecutedCommandLog
                resolved = _resolve_type_from_string(first)
                if resolved is not None:
                    return resolved
            return first if isinstance(first, type) else None
        return None

    if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
        non_none_types = _extract_union_types(field_type)
        if non_none_types:
            # If the first extracted type is a string, try to resolve it
            first = non_none_types[0]
            if isinstance(first, str):
                if first == "ExecutedCommandLog":
                    return ExecutedCommandLog
                resolved = _resolve_type_from_string(first)
                if resolved is not None:
                    return resolved
            if isinstance(first, type):
                return first
            return None
        return None

    return field_type if isinstance(field_type, type) else None


def _annotation_allows_none(field_type: object) -> bool:
    try:
        return type(None) in get_args(field_type)
    except Exception:
        return False


def _deserialize_value(
    value: object, field_type: object, _context_model: type[BaseModel]
) -> object:
    """
    Deserialize a value according to its field type.

    This centralizes the deserialization logic and integrates with
    Flujo's serialization system.
    """
    _ = _context_model
    if field_type is None:
        return value

    origin = getattr(field_type, "__origin__", None)
    if origin is list and isinstance(value, list):
        element_args = get_args(field_type)
        element_type = element_args[0] if element_args else None
        resolved_element_type = _resolve_actual_type(element_type) or element_type
        if isinstance(resolved_element_type, type):
            # Handle Pydantic/Flujo BaseModel items
            if (
                hasattr(resolved_element_type, "model_validate")
                and callable(getattr(resolved_element_type, "model_validate", None))
                and (
                    issubclass(resolved_element_type, BaseModel)
                    or issubclass(resolved_element_type, PydanticBaseModel)
                )
            ):
                deserialized: list[object] = []
                for item in value:
                    if isinstance(item, resolved_element_type):
                        deserialized.append(item)
                    elif isinstance(item, dict):
                        try:
                            deserialized.append(resolved_element_type.model_validate(item))
                            continue
                        except Exception:
                            deserialized.append(item)
                    else:
                        deserialized.append(item)
                return deserialized

            # Handle custom deserializers for list elements
            from flujo.utils.serialization import lookup_custom_deserializer

            custom_deserializer = lookup_custom_deserializer(resolved_element_type)
            if custom_deserializer:
                try:
                    return [custom_deserializer(item) for item in value]
                except Exception:
                    return value
        return value

    if isinstance(value, dict):
        resolved_type = _resolve_actual_type(field_type) or field_type
        if isinstance(resolved_type, type):
            if (
                hasattr(resolved_type, "model_validate")
                and callable(getattr(resolved_type, "model_validate", None))
                and (
                    issubclass(resolved_type, BaseModel)
                    or issubclass(resolved_type, PydanticBaseModel)
                )
            ):
                try:
                    return resolved_type.model_validate(value)
                except Exception:
                    return value

            from flujo.utils.serialization import lookup_custom_deserializer

            custom_deserializer = lookup_custom_deserializer(resolved_type)
            if custom_deserializer:
                try:
                    return custom_deserializer(value)
                except Exception:
                    return value

    return value


def _build_context_update(
    output: BaseModel | JSONObject | object,
) -> JSONObject | None:
    """Return context update dict extracted from a step output."""
    if isinstance(output, (BaseModel, PydanticBaseModel)):
        # Handle PipelineResult objects from as_step
        # Important: use full dump (exclude_unset=False) so in-place mutations
        # to lists/dicts (e.g., command_log) are preserved.
        if hasattr(output, "final_pipeline_context") and output.final_pipeline_context is not None:
            result = output.final_pipeline_context.model_dump(exclude_unset=False)
            return result if isinstance(result, dict) else None
        # Handle regular BaseModel objects
        result = output.model_dump(exclude_unset=True, exclude_none=False)
        return result if isinstance(result, dict) else None
    if isinstance(output, dict):
        return output
    return None


def _deep_merge_dicts(base: JSONObject, update: JSONObject) -> JSONObject:
    """Deep merge update dict into base dict, handling nested structures."""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Both are dicts, recursively merge
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # Not both dicts, or key doesn't exist in base, overwrite
            result[key] = value

    return result


def _inject_context_with_deep_merge(
    context: BaseModel,
    update_data: JSONObject,
    context_model: type[BaseModel],
) -> Optional[str]:
    """Apply ``update_data`` to ``context`` with deep merge for nested dicts.

    Returns an error message if validation fails, otherwise ``None``.
    """
    if update_contains_scratchpad(update_data):
        return SCRATCHPAD_REMOVED_MESSAGE
    # Micro-optimization fast path for hot paths:
    # If the update only contains a few scalar fields that exist on the model and
    # types match exactly, assign them directly without performing a full
    # model_dump/validation/deep-merge cycle. This preserves type safety while
    # reducing overhead in performance tests.
    try:
        if update_data and len(update_data) <= 4:
            simple_update = True
            for key, value in update_data.items():
                if key not in context_model.model_fields:
                    simple_update = False
                    break
                field_info = context_model.model_fields[key]
                field_type = field_info.annotation
                try:
                    resolved = _resolve_actual_type(field_type)
                    if resolved is not None:
                        field_type = resolved
                except Exception:
                    pass
                # Disallow containers/complex types for the fast path
                try:
                    has_origin = hasattr(field_type, "__origin__")
                    if has_origin and getattr(field_type, "__origin__") in (list, dict):
                        simple_update = False
                        break
                except Exception:
                    simple_update = False
                    break
                if field_type not in (int, float, str, bool):
                    simple_update = False
                    break
                try:
                    # Guard against non-type annotations that make isinstance unreliable
                    if isinstance(field_type, type):
                        if value is None:
                            continue
                        if not isinstance(value, field_type):
                            simple_update = False
                            break
                    else:
                        simple_update = False
                        break
                except Exception:
                    simple_update = False
                    break
            if simple_update:
                # Capture original state for safe rollback on validation failure
                _original_fast = context.model_dump()
                for key, value in update_data.items():
                    setattr(context, key, value)
                # Final validation to enforce model-level invariants
                try:
                    validated = context_model.model_validate(context.model_dump())
                    context.__dict__.update(validated.__dict__)
                    return None
                except ValidationError as e:
                    for k, v in _original_fast.items():
                        try:
                            setattr(context, k, v)
                        except Exception:
                            try:
                                object.__setattr__(context, k, v)
                            except Exception:
                                # Best-effort rollback; continue restoring other fields
                                pass
                    return str(e)
    except Exception:
        # Fall through to the general path on any error
        pass

    original = context.model_dump()

    # Lenient fast-path for PipelineContext-style updates coming from as_step
    try:
        import os as _os

        _lenient_flag = str(_os.environ.get("FLUJO_LENIENT_AS_STEP_CONTEXT", "1")).strip().lower()
        _lenient_enabled = _lenient_flag in {"1", "true", "yes", "on"}
        if _lenient_enabled and any(k in update_data for k in ("command_log", "hitl_history")):
            for key, value in update_data.items():
                if key not in context_model.model_fields:
                    continue
                current_val = getattr(context, key, None)
                if isinstance(current_val, dict) and isinstance(value, dict):
                    try:
                        current_val.update(value)
                    except Exception:
                        setattr(context, key, value)
                else:
                    # For list-typed fields, deserialize elements to proper model types
                    field_info = context_model.model_fields[key]
                    field_type = field_info.annotation
                    try:
                        resolved = _resolve_actual_type(field_type)
                        if resolved is not None:
                            field_type = resolved
                    except Exception:
                        pass
                    try:
                        if (
                            field_type is not None
                            and hasattr(field_type, "__origin__")
                            and field_type.__origin__ is list
                            and isinstance(value, list)
                        ):
                            value = _deserialize_value(value, field_type, context_model)
                    except Exception:
                        pass
                    setattr(context, key, value)
            return None
    except Exception:
        # Fall through to validated path on any error
        pass

    # Process update data with proper field mapping
    for key, value in update_data.items():
        if key in context_model.model_fields:
            field_info = context_model.model_fields[key]
            raw_field_type = field_info.annotation
            field_type = raw_field_type
            # Resolve Union and other composite annotations to a concrete type when possible
            if value is None and _annotation_allows_none(raw_field_type):
                setattr(context, key, None)
                continue
            try:
                resolved = _resolve_actual_type(field_type)
                if resolved is not None:
                    field_type = resolved
            except Exception:
                pass

            # Special-case common textual fields to avoid over-validation
            if key == "initial_prompt" and isinstance(value, str):
                try:
                    setattr(context, key, value)
                    continue
                except Exception:
                    # Fall back to generic path
                    pass

            # TYPE VALIDATION: Ensure the value matches the declared field type
            if field_type is not None:
                try:
                    if value is None:
                        # Let Pydantic handle Optional/required semantics in the final validation pass.
                        pass
                    # For list fields, ensure we're not trying to assign a dict to a list[int]
                    elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                        if not isinstance(value, list):
                            return f"Field '{key}' expects list but got {type(value).__name__}: {value}"

                    # For int fields, ensure we're not trying to assign a dict to an int
                    elif field_type is int and not isinstance(value, int):
                        return f"Field '{key}' expects int but got {type(value).__name__}: {value}"

                    # For str fields, ensure we're not trying to assign a dict to a str
                    elif field_type is str and not isinstance(value, str):
                        return f"Field '{key}' expects str but got {type(value).__name__}: {value}"

                    # For dict fields, allow dict values
                    elif field_type is dict or (
                        hasattr(field_type, "__origin__") and field_type.__origin__ is dict
                    ):
                        if not isinstance(value, dict):
                            return f"Field '{key}' expects dict but got {type(value).__name__}: {value}"

                    # ImportArtifacts: coerce dicts into the typed container
                    elif field_type is ImportArtifacts:
                        if isinstance(value, dict):
                            value = ImportArtifacts.model_validate(value)
                        elif not isinstance(value, ImportArtifacts):
                            return (
                                f"Field '{key}' expects ImportArtifacts but got "
                                f"{type(value).__name__}: {value}"
                            )

                    # For other types, use Pydantic's validation
                    else:
                        # Try to validate the value against the field type.
                        # Be lenient for unresolved typing constructs (e.g., typing.Union).
                        try:
                            ft_str = str(field_type)
                            if ft_str.startswith("typing."):
                                # Skip strict callable validation for typing.* constructs
                                pass
                            elif (
                                isinstance(field_type, type)
                                and hasattr(field_type, "model_validate")
                                and callable(getattr(field_type, "model_validate", None))
                                and isinstance(value, dict)
                            ):
                                value = field_type.model_validate(value)
                            elif (
                                isinstance(field_type, type)
                                and hasattr(field_type, "model_validate")
                                and callable(getattr(field_type, "model_validate", None))
                            ):
                                field_type.model_validate(value)
                        except Exception as validation_error:
                            return f"Field '{key}' validation failed: {validation_error}"

                except Exception as type_check_error:
                    return f"Field '{key}' type check failed: {type_check_error}"

            # Apply the validated value to the specific field

            if key == "import_artifacts":
                try:
                    if isinstance(value, dict):
                        value = ImportArtifacts.model_validate(value)
                    existing_artifacts = (
                        getattr(context, "import_artifacts", None)
                        if hasattr(context, "import_artifacts")
                        else None
                    )
                    if existing_artifacts is not None and not _is_import_artifacts(
                        existing_artifacts
                    ):
                        try:
                            existing_artifacts = ImportArtifacts.model_validate(
                                dict(existing_artifacts)
                            )
                        except Exception:
                            existing_artifacts = ImportArtifacts(**dict(existing_artifacts))
                        setattr(context, "import_artifacts", existing_artifacts)
                    if isinstance(existing_artifacts, MutableMapping):
                        if isinstance(value, MutableMapping):
                            existing_artifacts.update(value)
                            value = existing_artifacts
                        else:
                            value = existing_artifacts
                except Exception:
                    pass

            # For list-typed fields, deserialize elements into declared model types
            try:
                if (
                    field_type is not None
                    and hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and isinstance(value, list)
                ):
                    value = _deserialize_value(value, field_type, context_model)
            except Exception:
                pass
            setattr(context, key, value)
        else:
            # Allow dynamic or previously-added attributes (Pydantic BaseModel blocks setattr)
            if key == "steps":
                # Migrate to step_outputs
                try:
                    if hasattr(context, "step_outputs"):
                        current_outputs = getattr(context, "step_outputs", {}) or {}
                        if not isinstance(current_outputs, dict):
                            current_outputs = {}

                        # Normalize value
                        val_dict = value if isinstance(value, dict) else {"value": value}

                        # Merge
                        merged = _deep_merge_dicts(current_outputs, val_dict)
                        context.step_outputs = merged
                except Exception:
                    pass
                # Always skip setting context.steps directly (it is read-only)
                continue
            try:
                object.__setattr__(context, key, value)
            except Exception:
                setattr(context, key, value)

    # Final validation pass
    try:
        validated = context_model.model_validate(context.model_dump())
        context.__dict__.update(validated.__dict__)
    except ValidationError as e:
        # If validation fails, restore original state
        for key, value in original.items():
            setattr(context, key, value)
        telemetry.logfire.error(f"Context update failed Pydantic validation: {e}")
        return str(e)

    return None


def _inject_context(
    context: BaseModel,
    update_data: JSONObject,
    context_model: type[BaseModel],
) -> Optional[str]:
    """Apply ``update_data`` to ``context`` validating against ``context_model``.

    Returns an error message if validation fails, otherwise ``None``.
    """
    if update_contains_scratchpad(update_data):
        return SCRATCHPAD_REMOVED_MESSAGE
    original = context.model_dump()

    # Process update data with proper field mapping
    for key, value in update_data.items():
        if key in context_model.model_fields:
            field_info = context_model.model_fields[key]
            raw_field_type = field_info.annotation
            field_type = raw_field_type
            if value is None and _annotation_allows_none(raw_field_type):
                setattr(context, key, None)
                continue
            try:
                resolved = _resolve_actual_type(field_type)
                if resolved is not None:
                    field_type = resolved
            except Exception:
                pass

            # Deserialize dict/list payloads into declared model types before assignment.
            # This prevents transient "model field contains dict" states during `model_dump()`,
            # which can trigger Pydantic serializer warnings for model-typed fields.
            try:
                value = _deserialize_value(value, field_info.annotation, context_model)
            except Exception:
                pass

            # TYPE VALIDATION: Ensure the value matches the declared field type
            if field_type is not None:
                try:
                    # For list fields, ensure we're not trying to assign a dict to a list[int]
                    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                        if not isinstance(value, list):
                            return f"Field '{key}' expects list but got {type(value).__name__}: {value}"

                    # For int fields, ensure we're not trying to assign a dict to an int
                    elif field_type is int and not isinstance(value, int):
                        return f"Field '{key}' expects int but got {type(value).__name__}: {value}"

                    # For str fields, ensure we're not trying to assign a dict to a str
                    elif field_type is str and not isinstance(value, str):
                        return f"Field '{key}' expects str but got {type(value).__name__}: {value}"

                    # For dict fields, allow dict values
                    elif field_type is dict or (
                        hasattr(field_type, "__origin__") and field_type.__origin__ is dict
                    ):
                        if not isinstance(value, dict):
                            return f"Field '{key}' expects dict but got {type(value).__name__}: {value}"

                    # For other types, use Pydantic's validation
                    else:
                        # Try to validate the value against the field type
                        try:
                            if (
                                isinstance(field_type, type)
                                and hasattr(field_type, "model_validate")
                                and callable(getattr(field_type, "model_validate", None))
                            ):
                                if isinstance(value, dict):
                                    value = field_type.model_validate(value)
                                elif not isinstance(value, field_type):
                                    field_type.model_validate(value)
                        except Exception as validation_error:
                            return f"Field '{key}' validation failed: {validation_error}"

                except Exception as type_check_error:
                    return f"Field '{key}' type check failed: {type_check_error}"

            # Apply the validated value to the specific field
            setattr(context, key, value)
        else:
            try:
                object.__setattr__(context, key, value)
            except Exception:
                setattr(context, key, value)

    # Final validation pass
    try:
        validated = context_model.model_validate(context.model_dump())
        context.__dict__.update(validated.__dict__)
    except ValidationError as e:
        # If validation fails, restore original state
        for key, value in original.items():
            setattr(context, key, value)
        telemetry.logfire.error(f"Context update failed Pydantic validation: {e}")
        return str(e)

    return None
