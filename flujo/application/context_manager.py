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
    "_should_pass_context",
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
            # Check if the **kwargs parameter is typed as 'Never' (doesn't accept any kwargs)
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    # If the **kwargs parameter is annotated as 'Never', it doesn't accept any parameters
                    # Use direct comparison instead of string comparison for robustness
                    from typing import Never

                    if p.annotation is Never:
                        result = False
                    else:
                        result = True
                    break
            # If we didn't find a VAR_KEYWORD parameter, this should never happen
            # since we already checked for it in the elif condition
            else:
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


def _should_pass_context(spec: Any, context: Optional[Any], func: Callable[..., Any]) -> bool:
    """Determine if context should be passed to a function based on signature analysis.

    Args:
        spec: Signature analysis result from analyze_signature()
        context: The context object to potentially pass
        func: The function to analyze

    Returns:
        True if context should be passed to the function, False otherwise
    """
    # Check if function accepts context parameter (either explicitly or via **kwargs)
    # This is different from spec.needs_context which only checks if context is required
    # Compile-time check: spec.needs_context indicates whether the function explicitly requires
    # a `context` parameter based on signature analysis.
    # Runtime check: context is not None ensures that a context object is available, and
    # bool(accepts_context) verifies if the function can dynamically accept the context parameter.
    accepts_context = _accepts_param(func, "context")
    return spec.needs_context or (context is not None and bool(accepts_context))


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

    # Handle Tuple types - tuple is compatible with Tuple[...]
    if origin_b is tuple:
        # If b is Tuple[...], then any tuple type is compatible
        return a is tuple or (isinstance(a, type) and issubclass(a, tuple))
    if origin_a is tuple:
        # If a is Tuple[...], then it's compatible with tuple
        return b is tuple or (isinstance(b, type) and issubclass(b, tuple))

    # Handle Dict types - dict is compatible with Dict[...]
    if origin_b is dict:
        # If b is Dict[...], then any dict type is compatible
        return a is dict or (isinstance(a, type) and issubclass(a, dict))
    if origin_a is dict:
        # If a is Dict[...], then it's compatible with dict
        return b is dict or (isinstance(b, type) and issubclass(b, dict))

    # Only call issubclass if both are actual classes
    if not isinstance(a, type) or not isinstance(b, type):
        return False
    try:
        return issubclass(a, b)
    except Exception:
        return False
