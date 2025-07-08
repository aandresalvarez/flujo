from __future__ import annotations

import inspect
import weakref
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin
import types

from ..domain.dsl.step import Step
from ..domain.models import StepResult

__all__ = [
    "_accepts_param",
    "_extract_missing_fields",
    "_get_validation_flags",
    "_apply_validation_metadata",
    "_types_compatible",
]

_accepts_param_cache_weak: weakref.WeakKeyDictionary[
    Callable[..., Any], Dict[str, Optional[bool]]
] = weakref.WeakKeyDictionary()
_accepts_param_cache_id: weakref.WeakValueDictionary[int, Dict[str, Optional[bool]]] = (
    weakref.WeakValueDictionary()
)


def _get_validation_flags(step: Step[Any, Any]) -> tuple[bool, bool]:
    """Return (is_validation_step, is_strict) flags from step metadata."""
    is_validation_step = bool(step.meta.get("is_validation_step", False))
    is_strict = bool(step.meta.get("strict_validation", False)) if is_validation_step else False
    return is_validation_step, is_strict


def _apply_validation_metadata(
    result: StepResult,
    *,
    validation_failed: bool,
    is_validation_step: bool,
    is_strict: bool,
) -> None:
    """Set result metadata when non-strict validation fails."""
    if validation_failed and is_validation_step and not is_strict:
        result.metadata_ = result.metadata_ or {}
        result.metadata_["validation_passed"] = False


def _accepts_param(func: Callable[..., Any], param: str) -> Optional[bool]:
    """Return True if callable's signature includes ``param`` or ``**kwargs``."""
    try:
        cache = _accepts_param_cache_weak.setdefault(func, {})
    except TypeError:  # For unhashable callables
        func_id = id(func)
        cache = _accepts_param_cache_id.setdefault(func_id, {})
    if param in cache:
        return cache[param]

    try:
        sig = inspect.signature(func)
        if param in sig.parameters:
            result = True
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            result = True
        else:
            result = False
    except (TypeError, ValueError):
        result = None

    cache[param] = result
    return result


def _extract_missing_fields(cause: Any) -> list[str]:
    """Return list of missing field names from a Pydantic ValidationError."""
    missing_fields: list[str] = []
    if cause is not None and hasattr(cause, "errors"):
        for err in cause.errors():
            if err.get("type") == "missing":
                loc = err.get("loc") or []
                if isinstance(loc, (list, tuple)) and loc:
                    field = loc[0]
                    if isinstance(field, str):
                        missing_fields.append(field)
    return missing_fields


def _types_compatible(a: Any, b: Any) -> bool:
    """Return ``True`` if type ``a`` is compatible with type ``b``."""
    # If a is a value, get its type
    if not isinstance(a, type):
        a = type(a)
    if not isinstance(b, type) and get_origin(b) is None:
        b = type(b)

    if a is Any or b is Any:
        return True

    origin_a, origin_b = get_origin(a), get_origin(b)
    # Handle typing.Union and types.UnionType (Python 3.10+)
    if origin_b is Union:
        return any(_types_compatible(a, arg) for arg in get_args(b))
    if hasattr(types, "UnionType") and isinstance(b, types.UnionType):
        return any(_types_compatible(a, arg) for arg in b.__args__)
    if origin_a is Union:
        return all(_types_compatible(arg, b) for arg in get_args(a))
    if hasattr(types, "UnionType") and isinstance(a, types.UnionType):
        return all(_types_compatible(arg, b) for arg in a.__args__)

    # Only call issubclass if both are actual classes
    if not isinstance(a, type) or not isinstance(b, type):
        return False
    try:
        return issubclass(a, b)
    except Exception:
        return False
