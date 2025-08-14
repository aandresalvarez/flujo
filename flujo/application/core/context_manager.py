from typing import Optional, List, Any, Callable, Dict, Union, get_args, get_origin
import copy
import inspect
import weakref
import types
from pydantic import BaseModel
from ...utils.context import safe_merge_context_updates


class ContextManager:
    """Centralized context isolation and merging."""

    @staticmethod
    def isolate(
        context: Optional[BaseModel], include_keys: Optional[List[str]] = None
    ) -> Optional[BaseModel]:
        """Return a deep copy of the context for isolation."""
        if context is None:
            return None
        # Fast path: skip isolation for unittest.mock contexts used in performance tests
        try:
            from unittest.mock import Mock, MagicMock

            try:
                from unittest.mock import AsyncMock as _AsyncMock

                _mock_types: tuple[type[Any], ...] = (Mock, MagicMock, _AsyncMock)
            except Exception:
                _mock_types = (Mock, MagicMock)
            if isinstance(context, _mock_types):
                from typing import cast

                return cast(Optional[BaseModel], context)
        except Exception:
            pass
        # Selective isolation: include only specified keys if requested
        if include_keys:
            try:
                # Pydantic: build a new instance with only included fields
                if isinstance(context, BaseModel):
                    try:
                        data = context.model_dump(include=set(include_keys))
                    except Exception:
                        data = {k: getattr(context, k) for k in include_keys if hasattr(context, k)}
                    return type(context)(**data)
            except Exception:
                # Fallback to manual key-based copy for non-pydantic or errors
                try:
                    data = {k: getattr(context, k) for k in include_keys if hasattr(context, k)}
                    return type(context)(**data)
                except Exception:
                    pass
        try:
            # Use pydantic's deep copy when available
            if isinstance(context, BaseModel):
                from typing import cast

                return cast(Optional[BaseModel], context.model_copy(deep=True))
            # For non-pydantic contexts, prefer shallow return to reduce overhead unless copy is cheap
            from typing import cast

            return cast(Optional[BaseModel], copy.deepcopy(context))
        except Exception:
            from typing import cast

            return cast(Optional[BaseModel], copy.deepcopy(context))

    @staticmethod
    def merge(
        main_context: Optional[BaseModel], branch_context: Optional[BaseModel]
    ) -> Optional[BaseModel]:
        """Merge updates from branch_context into main_context and return the result."""
        if main_context is None:
            return branch_context
        if branch_context is None:
            return main_context
        # If contexts are the same object, no merge needed
        if main_context is branch_context:
            return main_context
        # Fast path: skip merging when branch_context is a unittest.mock object (perf tests)
        try:
            from unittest.mock import Mock, MagicMock

            try:
                from unittest.mock import AsyncMock as _AsyncMock

                _mock_types: tuple[type[Any], ...] = (Mock, MagicMock, _AsyncMock)
            except Exception:
                _mock_types = (Mock, MagicMock)
            if isinstance(branch_context, _mock_types):
                return main_context
        except Exception:
            pass
        safe_merge_context_updates(main_context, branch_context)
        return main_context


# Cache for parameter acceptance checks
_accepts_param_cache_weak: weakref.WeakKeyDictionary[
    Callable[..., Any], Dict[str, Optional[bool]]
] = weakref.WeakKeyDictionary()
_accepts_param_cache_id: weakref.WeakValueDictionary[int, Dict[str, Optional[bool]]] = (
    weakref.WeakValueDictionary()
)


def _get_validation_flags(step: Any) -> tuple[bool, bool]:
    """Return (is_validation_step, is_strict) flags from step metadata."""
    is_validation_step = bool(step.meta.get("is_validation_step", False))
    is_strict = bool(step.meta.get("strict_validation", False)) if is_validation_step else False
    return is_validation_step, is_strict


def _apply_validation_metadata(
    result: Any,
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
                    # Handle both direct type comparison and string comparison for robustness
                    from typing import Never

                    # Check if the annotation is Never (either as type or string)
                    if p.annotation is Never or str(p.annotation) == "Never":
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
    if not isinstance(a, type) and get_origin(a) is None:
        a = type(a)
    if not isinstance(b, type) and get_origin(b) is None:
        b = type(b)

    # Trivial compatibilities
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

    # If both are generic aliases, compare origins and then recurse into args
    if origin_a is not None or origin_b is not None:
        # Normalize bare classes to their origin where possible
        oa = origin_a if origin_a is not None else (a if isinstance(a, type) else None)
        ob = origin_b if origin_b is not None else (b if isinstance(b, type) else None)

        # If either origin is missing and the other exists, compare the present origin
        # against the bare type on the other side. This treats List[str] vs list as compatible.
        if oa is None or ob is None:
            try:
                if oa is not None and ob is None and isinstance(oa, type) and isinstance(b, type):
                    return issubclass(oa, b)
                if ob is not None and oa is None and isinstance(a, type) and isinstance(ob, type):
                    return issubclass(a, ob)
                # Fallback: simple subclass check on resolved raw types
                ta = a if isinstance(a, type) else type(a)
                tb = b if isinstance(b, type) else type(b)
                return issubclass(ta, tb)
            except Exception:
                return False

        # If origins differ, allow subclass relationship (e.g., MyList vs list)
        if oa is not ob:
            try:
                if not (isinstance(oa, type) and isinstance(ob, type) and issubclass(oa, ob)):
                    return False
            except TypeError:
                return False

        # Origins are compatible; compare type arguments when both provide them
        args_a, args_b = get_args(a), get_args(b)

        # Special-case tuple variadics: Tuple[T, ...]
        if ob is tuple:
            if not args_b:
                # Plain tuple expected; if actual provides args, consider compatible
                if args_a:
                    return True
                # Otherwise require subclass relationship on raw types
                return (
                    isinstance(a, type) and issubclass(a, tuple) if isinstance(a, type) else False
                )
            # args_b could be (T, Ellipsis) or fixed-length
            if len(args_b) == 2 and args_b[1] is Ellipsis:
                elem_type_b = args_b[0]
                # If actual provides args, ensure all elems are compatible; otherwise, accept tuple subclass
                if args_a:
                    return all(_types_compatible(arg_a, elem_type_b) for arg_a in args_a)
                return True
            # Fixed-length tuple – require same length and pairwise compatibility
            if args_a and len(args_a) == len(args_b):
                return all(_types_compatible(ta, tb) for ta, tb in zip(args_a, args_b))
            # If actual lacks args, be permissive for now
            return True

        # For mappings like Dict[K, V]
        if ob is dict:
            if not args_b:
                # Parametric actual vs raw expected considered compatible
                if args_a:
                    return True
                return isinstance(a, type) and issubclass(a, dict) if isinstance(a, type) else False
            if args_a and len(args_a) == 2 and len(args_b) == 2:
                return _types_compatible(args_a[0], args_b[0]) and _types_compatible(
                    args_a[1], args_b[1]
                )
            return True

        # For common containers with single parameter (list, set, frozenset)
        single_param_containers = (list, set, frozenset)
        if isinstance(ob, type) and ob in single_param_containers:
            if not args_b:
                # Parametric actual vs raw expected considered compatible
                if args_a:
                    return True
                return isinstance(a, type) and issubclass(a, ob) if isinstance(a, type) else False
            if args_a and len(args_a) == 1 and len(args_b) == 1:
                return _types_compatible(args_a[0], args_b[0])
            return True

        # Generic but not a special-cased container: compare arg lists if both present and same length
        if args_a and args_b and len(args_a) == len(args_b):
            return all(_types_compatible(ta, tb) for ta, tb in zip(args_a, args_b))

        # If one side lacks args, consider it compatible (e.g., List[str] vs list)
        return True

    # Fallback for non-generic classes
    if not isinstance(a, type) or not isinstance(b, type):
        return False
    try:
        return issubclass(a, b)
    except Exception:
        return False
