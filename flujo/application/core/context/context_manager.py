from typing import Callable, TypeGuard, Union, get_args, get_origin
import copy
import inspect
import os
import weakref
import types
import logging
from ....domain.models import BaseModel
from ....infra import telemetry
from ....utils.context import safe_merge_context_updates
from ....exceptions import ContextMutationError
from .context_strategies import (
    ContextIsolationStrategy,
    LenientIsolation,
    StrictIsolation,
    SelectiveIsolation,
)

try:
    from ....infra.settings import get_settings as _get_settings_default

    _SETTINGS_DEFAULTS: object | None = _get_settings_default()
except Exception:
    _SETTINGS_DEFAULTS = None


class ContextManager:
    """Centralized context isolation and merging.

    Phase 1 Enhancement: Added strict mode for mutation detection.
    Set FLUJO_STRICT_CONTEXT=1 to enable runtime mutation verification.
    """

    # Module-level flag for strict context mutation detection (Phase 1)
    ENFORCE_ISOLATION: bool = os.environ.get("FLUJO_STRICT_CONTEXT", "0") == "1"

    @staticmethod
    def _is_base_model(obj: object) -> TypeGuard[BaseModel]:
        return isinstance(obj, BaseModel)

    class ContextIsolationError(Exception):
        """Raised when context isolation fails under strict settings."""

        pass

    class ContextMergeError(Exception):
        """Raised when context merging fails under strict settings."""

        pass

    @staticmethod
    def _resolve_strategy(
        include_keys: list[str] | None = None,
    ) -> ContextIsolationStrategy:
        """Select an isolation strategy based on settings and include_keys."""
        strict_iso = False
        strict_merge = False
        settings_obj: object | None = _SETTINGS_DEFAULTS
        if settings_obj is None:
            try:
                # Fallback: fetch settings lazily to support tests that monkeypatch env
                from ....infra.settings import get_settings as _gs  # local import to avoid cycles

                settings_obj = _gs()
            except Exception:
                settings_obj = None
        try:
            strict_iso = bool(getattr(settings_obj, "strict_context_isolation", False))
            strict_merge = bool(getattr(settings_obj, "strict_context_merge", False))
        except Exception:
            strict_iso = False
            strict_merge = False

        # Selective isolation path preserves explicit include_keys even under lenient settings
        if include_keys:
            iso_impl = (
                ContextManager._isolate_strict_impl
                if strict_iso
                else ContextManager._isolate_lenient_impl
            )
            merge_impl = (
                ContextManager._merge_strict_impl
                if strict_merge
                else ContextManager._merge_lenient_impl
            )
            return SelectiveIsolation(iso_impl, merge_impl, include_keys=list(include_keys))

        if strict_iso or strict_merge:
            iso_impl = (
                ContextManager._isolate_strict_impl
                if strict_iso
                else ContextManager._isolate_lenient_impl
            )
            merge_impl = (
                ContextManager._merge_strict_impl
                if strict_merge
                else ContextManager._merge_lenient_impl
            )
            return StrictIsolation(iso_impl, merge_impl)

        return LenientIsolation(
            ContextManager._isolate_lenient_impl, ContextManager._merge_lenient_impl
        )

    @staticmethod
    def _isolate_lenient_impl(
        context: BaseModel | None, include_keys: list[str] | None = None
    ) -> BaseModel | None:
        """Return a deep copy of the context for isolation (lenient behavior)."""
        if context is None:
            return None
        # Selective isolation: include only specified keys if requested
        if include_keys:
            try:
                # Pydantic: build a new instance with only included fields
                if isinstance(context, BaseModel):
                    model_cls = type(context)
                    try:
                        # Prefer explicit field map (Pydantic v2)
                        fields = getattr(model_cls, "model_fields", None)
                        if isinstance(fields, dict) and fields:
                            new_data: dict[str, object] = {}
                            for name, field in fields.items():
                                if name in include_keys:
                                    # Preserve included fields from original context
                                    new_data[name] = getattr(context, name, None)
                                else:
                                    # For excluded fields, try to restore defaults; if none, keep original
                                    if (
                                        getattr(field, "default", inspect._empty)
                                        is not inspect._empty
                                    ):
                                        new_data[name] = field.default
                                    elif getattr(field, "default_factory", None) is not None:
                                        try:
                                            new_data[name] = field.default_factory()
                                        except Exception:
                                            new_data[name] = getattr(context, name, None)
                                    else:
                                        # Required without default: use original value to avoid construction errors
                                        new_data[name] = getattr(context, name, None)
                            return model_cls(**new_data)

                        # Fallback for Pydantic v1 (__fields__)
                        fields_v1 = getattr(model_cls, "__fields__", None)
                        if isinstance(fields_v1, dict) and fields_v1:
                            new_data_v1: dict[str, object] = {}
                            for name, field in fields_v1.items():
                                if name in include_keys:
                                    new_data_v1[name] = getattr(context, name, None)
                                else:
                                    # v1: prefer default/default_factory; else keep original value
                                    default = getattr(field, "default", inspect._empty)
                                    if default is not inspect._empty:
                                        new_data_v1[name] = default
                                    elif getattr(field, "default_factory", None) is not None:
                                        try:
                                            new_data_v1[name] = field.default_factory()
                                        except Exception:
                                            new_data_v1[name] = getattr(context, name, None)
                                    else:
                                        new_data_v1[name] = getattr(context, name, None)
                            return model_cls(**new_data_v1)

                        # Last resort: use model_dump include, then construct
                        data = context.model_dump(include=set(include_keys))
                        return model_cls(**data)
                    except Exception:
                        # Fall through to generic handling below
                        pass
            except Exception:
                # Fallback to manual key-based copy for non-pydantic or errors
                try:
                    data = {k: getattr(context, k) for k in include_keys if hasattr(context, k)}
                    return type(context)(**data)
                except Exception:
                    pass
        # Use pydantic's deep copy when available; otherwise fall back safely
        if ContextManager._is_base_model(context):
            try:
                return context.model_copy(deep=True)
            except Exception:
                # Fall through to generic clone
                pass

        return _clone_context(context)

    @staticmethod
    def isolate_strict(
        context: BaseModel | None, include_keys: list[str] | None = None
    ) -> BaseModel | None:
        """Isolate context strictly: raise if deep isolation cannot be guaranteed.

        - For Pydantic, uses model_copy(deep=True).
        - Else, attempts deepcopy. If both fail, raises ContextIsolationError.
        """
        return ContextManager._isolate_strict_impl(context, include_keys)

    @staticmethod
    def isolate(
        context: BaseModel | None,
        include_keys: list[str] | None = None,
        *,
        purpose: str = "unknown",
    ) -> BaseModel | None:
        """Isolate context using the configured strategy.

        Args:
            context: The context to isolate
            include_keys: Optional list of keys to include in isolation
            purpose: Purpose of isolation (for debugging in strict mode)

        Returns:
            Isolated context copy
        """
        strategy = ContextManager._resolve_strategy(include_keys)
        isolated = strategy.isolate(context, include_keys)

        # Phase 1: Track isolation for mutation detection (strict mode)
        if ContextManager.ENFORCE_ISOLATION and isolated is not None and context is not None:
            # Track original for mutation detection
            try:
                object.__setattr__(isolated, "_original_context_id", id(context))
                object.__setattr__(isolated, "_isolation_purpose", purpose)
            except Exception:
                # If we can't set attributes (e.g., frozen model), skip tracking
                pass

        return isolated

    @staticmethod
    def _isolate_strict_impl(
        context: BaseModel | None, include_keys: list[str] | None = None
    ) -> BaseModel | None:
        last_error: Exception | None = None
        if context is None:
            return None

        # Selective isolation if keys provided (reuse non-strict which is safe)
        if include_keys:
            isolated = ContextManager._isolate_lenient_impl(context, include_keys)
            if isolated is None:
                raise ContextManager.ContextIsolationError("Selective isolation returned None")
            return isolated

        # Pydantic deep copy preferred
        try:
            if ContextManager._is_base_model(context):
                isolated = context.model_copy(deep=True)
                return isolated
        except Exception as e:
            last_error = e

        try:
            cloned = _clone_context(context)
            if cloned is None:
                raise ContextManager.ContextIsolationError(
                    f"Deep isolation returned None for context type {type(context).__name__}"
                )
            if cloned is context:
                raise ContextManager.ContextIsolationError(
                    f"Strict isolation fell back to shared reference for context type "
                    f"{type(context).__name__}"
                )
            return cloned
        except Exception as e:
            raise ContextManager.ContextIsolationError(
                f"Deep isolation failed for context type {type(context).__name__}: {e or last_error}"
            ) from e

    @staticmethod
    def verify_isolation(
        original: BaseModel | None,
        isolated: BaseModel | None,
    ) -> None:
        """Verify that isolated context hasn't mutated original (strict mode).

        This method checks for shared references in common mutable fields.

        Args:
            original: The original context
            isolated: The isolated context to verify

        Raises:
            ContextMutationError: If mutation violation is detected
        """
        if not ContextManager.ENFORCE_ISOLATION:
            return
        if original is None or isolated is None:
            return

        from collections.abc import MutableMapping as _MutableMapping

        def _collect_mutables(ctx: BaseModel) -> dict[str, object]:
            out: dict[str, object] = {}
            for name in (
                "step_outputs",
                "hitl_history",
                "command_log",
                "conversation_history",
                "hitl_data",
                "import_artifacts",
            ):
                try:
                    val = getattr(ctx, name, None)
                    if isinstance(val, (dict, list, _MutableMapping)):
                        out[name] = val
                except Exception:
                    continue
            return out

        orig_mut = _collect_mutables(original)
        iso_mut = _collect_mutables(isolated)
        for name, orig_val in orig_mut.items():
            iso_val = iso_mut.get(name)
            if orig_val is iso_val and orig_val is not None:
                purpose = getattr(isolated, "_isolation_purpose", "unknown")
                raise ContextMutationError(
                    f"Isolated context shares '{name}' reference with original. "
                    f"Isolation purpose: {purpose}",
                    suggestion="Ensure context isolation deep-copies mutable fields",
                    code="CONTEXT_MUTATION_VIOLATION",
                )

        # Additional check: verify context IDs match tracking
        if hasattr(isolated, "_original_context_id"):
            tracked_id = getattr(isolated, "_original_context_id")
            if tracked_id != id(original):
                purpose = getattr(isolated, "_isolation_purpose", "unknown")
                raise ContextMutationError(
                    f"Isolated context tracking mismatch. Expected original id {id(original)}, "
                    f"got {tracked_id}. Isolation purpose: {purpose}",
                    suggestion="Context may have been replaced or recreated",
                    code="CONTEXT_TRACKING_MISMATCH",
                )

    @staticmethod
    def merge(main_context: BaseModel | None, branch_context: BaseModel | None) -> BaseModel | None:
        """Merge updates from branch_context into main_context using configured strategy."""
        strategy = ContextManager._resolve_strategy()
        merged = strategy.merge(main_context, branch_context)
        # ContextReference values are held in private attrs and can be dropped by merge
        # implementations that rebuild models (e.g., strict merge fallbacks).
        try:
            from ....domain.models import ContextReference

            if (
                isinstance(merged, BaseModel)
                and isinstance(main_context, BaseModel)
                and (branch_context is None or isinstance(branch_context, BaseModel))
            ):
                for field_name in type(merged).model_fields:
                    field_value = getattr(merged, field_name, None)
                    if not isinstance(field_value, ContextReference):
                        continue
                    src = None
                    if branch_context is not None and hasattr(branch_context, field_name):
                        cand = getattr(branch_context, field_name, None)
                        if isinstance(cand, ContextReference) and cand._value is not None:
                            src = cand
                    if src is None and hasattr(main_context, field_name):
                        cand = getattr(main_context, field_name, None)
                        if isinstance(cand, ContextReference) and cand._value is not None:
                            src = cand
                    if src is not None:
                        field_value.set(src._value)
        except (AttributeError, TypeError) as exc:
            telemetry.logfire.debug(
                f"[ContextManager] Failed to preserve ContextReference values: {type(exc).__name__}: {exc}"
            )
        return merged

    @staticmethod
    def merge_strict(
        main_context: BaseModel | None, branch_context: BaseModel | None
    ) -> BaseModel | None:
        """Strict merge with robust fallbacks and error signaling."""
        return ContextManager._merge_strict_impl(main_context, branch_context)

    @staticmethod
    def _merge_lenient_impl(
        main_context: BaseModel | None, branch_context: BaseModel | None
    ) -> BaseModel | None:
        """Merge updates from branch_context into main_context and return the result."""
        # NOTE: Removed thread check that was skipping merge on non-main threads.
        # This caused test failures in CI when running with pytest-xdist (parallel workers)
        # because xdist worker processes run tests in worker threads, not the main thread.
        # The original check was likely intended to prevent race conditions, but
        # context merging should work correctly regardless of thread context.
        if main_context is None:
            return branch_context
        if branch_context is None:
            return main_context
        # If contexts are the same object, no merge needed
        if main_context is branch_context:
            return main_context
        safe_merge_context_updates(main_context, branch_context)
        return main_context

    @staticmethod
    def _merge_strict_impl(
        main_context: BaseModel | None, branch_context: BaseModel | None
    ) -> BaseModel | None:
        """Strict merge with robust fallbacks and error signaling.

        - Tries safe_merge_context_updates first; if it raises, raise ContextMergeError.
        - If merge returns falsey or appears ineffective, performs attribute-wise merge on a
          shallow copy of main with branch fields taking precedence.
        - Raises ContextMergeError if shallow copy or manual merge fails.
        """
        if main_context is None and branch_context is None:
            return None
        if main_context is None:
            return branch_context
        if branch_context is None:
            return main_context
        if main_context is branch_context:
            return main_context

        try:
            ok = safe_merge_context_updates(main_context, branch_context)
            if ok:
                return main_context
        except Exception as e:
            raise ContextManager.ContextMergeError(f"safe_merge_context_updates failed: {e}") from e

        # Manual attribute-wise merge with branch precedence
        try:
            new_context = copy.copy(main_context)
        except Exception as copy_err:
            raise ContextManager.ContextMergeError(
                f"Shallow copy of main_context failed: {copy_err}"
            ) from copy_err

        def _public_attrs(obj: object) -> list[str]:
            try:
                data = vars(obj)
                if isinstance(data, dict):
                    return [k for k in data.keys() if not k.startswith("_")]
                return []
            except Exception:
                pass
            try:
                return [a for a in dir(obj) if not a.startswith("_")]
            except Exception:
                return []

        def _try_set(target: object, name: str, value: object) -> None:
            try:
                object.__setattr__(target, name, value)
                return
            except Exception:
                pass
            try:
                setattr(target, name, value)
            except Exception:
                return

        try:
            for name in _public_attrs(main_context):
                try:
                    _try_set(new_context, name, getattr(main_context, name))
                except Exception:
                    pass
            for name in _public_attrs(branch_context):
                try:
                    _try_set(new_context, name, getattr(branch_context, name))
                except Exception:
                    pass
            return new_context
        except Exception as e:
            raise ContextManager.ContextMergeError(f"Manual context merge failed: {e}") from e


# Cache for parameter acceptance checks. We only cache weak-ref-able callables to avoid
# id()-based collisions when Python reuses memory addresses after garbage collection.
_accepts_param_cache_weak: weakref.WeakKeyDictionary[
    Callable[..., object], dict[str, bool | None]
] = weakref.WeakKeyDictionary()


def _get_validation_flags(step: object) -> tuple[bool, bool]:
    """Return (is_validation_step, is_strict) flags from step metadata."""
    meta_obj = getattr(step, "meta", None)
    meta: dict[str, object] = meta_obj if isinstance(meta_obj, dict) else {}
    is_validation_step = bool(meta.get("is_validation_step", False))
    is_strict = bool(meta.get("strict_validation", False)) if is_validation_step else False
    return is_validation_step, is_strict


def _apply_validation_metadata(
    result: object,
    *,
    validation_failed: bool,
    is_validation_step: bool,
    is_strict: bool,
) -> None:
    """Set result metadata when non-strict validation fails."""
    if validation_failed and is_validation_step and not is_strict:
        metadata_obj = getattr(result, "metadata_", None)
        metadata: dict[str, object]
        if isinstance(metadata_obj, dict):
            metadata = metadata_obj
        else:
            metadata = {}
            try:
                setattr(result, "metadata_", metadata)
            except Exception:
                return
        metadata["validation_passed"] = False


def _accepts_param(func: Callable[..., object], param: str) -> bool | None:
    """Return True if callable's signature includes ``param`` or ``**kwargs``."""
    cache_key: Callable[..., object] = func
    try:
        if isinstance(func, types.MethodType):
            # Cache on the underlying function to avoid per-access bound-method objects
            # causing unbounded cache growth.
            cache_key = func.__func__
    except Exception:
        cache_key = func
    try:
        cache = _accepts_param_cache_weak.setdefault(cache_key, {})
    except TypeError:
        # Unhashable / non-weakref-able callables (e.g., bound methods) are not cached to
        # avoid id()-based collisions when memory addresses are reused.
        cache = None

    if cache is not None and param in cache:
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
                    try:
                        from typing import Never
                    except ImportError:
                        try:
                            from typing_extensions import Never
                        except ImportError:
                            from typing import NoReturn as Never

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

    if cache is not None:
        cache[param] = result
    return result


def _extract_missing_fields(cause: object) -> list[str]:
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


def _should_pass_context(spec: object, context: object | None, func: Callable[..., object]) -> bool:
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
    needs_context = bool(getattr(spec, "needs_context", False))
    return needs_context or (context is not None and bool(accepts_context))


def _clone_context(obj: BaseModel | None) -> BaseModel | None:
    """Efficient deep clone favoring Pydantic's model_copy over stdlib deepcopy."""
    log = logging.getLogger("flujo.context")
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        # Try fastest safe path
        try:
            return obj.model_copy(deep=True)
        except Exception:
            pass
        # Field-wise deepcopy with skip-on-failure to avoid shared mutable resources
        try:
            data: dict[str, object] = {}
            for name in type(obj).model_fields:
                val = getattr(obj, name, None)
                try:
                    data[name] = copy.deepcopy(val)
                except Exception as exc:
                    # Skip non-copyable field so default_factory/default can populate during validate
                    log.debug(
                        "Context isolation skipped non-copyable field %s on %s: %s",
                        name,
                        type(obj).__name__,
                        exc,
                    )
                    continue
            return type(obj).model_validate(data)
        except Exception:
            pass
    try:
        return copy.deepcopy(obj)
    except Exception:
        # Shallow copy as a last resort; log to surface potential shared-state risk.
        try:
            cloned = copy.copy(obj)
            if cloned is obj:
                log.warning(
                    "Context isolation fallback returned original object; potential shared state risk",
                    extra={"context_type": type(obj).__name__},
                )
            return cloned
        except Exception:
            log.warning(
                "Context isolation failed to copy object; returning original (shared) reference",
                extra={"context_type": type(obj).__name__},
            )
            return obj


def _types_compatible(a: object, b: object) -> bool:
    """Return ``True`` if type ``a`` is compatible with type ``b``."""

    # --- Helper utilities to keep the core logic readable ---
    def _is_typing_any(tp: object) -> bool:
        try:
            name = getattr(tp, "_name", None)
            if isinstance(name, str) and name.lower() == "any":
                return True
            rep = repr(tp)
            return isinstance(rep, str) and rep.lower() in {"typing.any", "any"}
        except Exception:
            return False

    def _is_union_type(tp: object) -> bool:
        try:
            origin = get_origin(tp)
            if origin is Union:
                return True
            # Python 3.10+ syntax: int | None
            return hasattr(types, "UnionType") and isinstance(tp, types.UnionType)
        except Exception:
            return False

    def _iter_union_args(tp: object) -> list[object]:
        try:
            origin = get_origin(tp)
            if origin is Union:
                return list(get_args(tp))
            if hasattr(types, "UnionType") and isinstance(tp, types.UnionType):
                return list(getattr(tp, "__args__", ()))
        except Exception:
            pass
        return []

    def _issubclass_safe(sub: object, sup: object) -> bool:
        try:
            return isinstance(sub, type) and isinstance(sup, type) and issubclass(sub, sup)
        except Exception:
            return False

    def _compare_tuple(a_args: tuple[object, ...], b_args: tuple[object, ...]) -> bool:
        # Tuple[T, ...] variadic form
        if len(b_args) == 2 and b_args[1] is Ellipsis:
            elem_b = b_args[0]
            if a_args:
                return all(_types_compatible(arg_a, elem_b) for arg_a in a_args)
            return True
        # Fixed-length tuples
        if a_args and len(a_args) == len(b_args):
            return all(_types_compatible(ta, tb) for ta, tb in zip(a_args, b_args, strict=True))
        # Be permissive if actual lacks args
        return True

    def _compare_dict(a_args: tuple[object, ...], b_args: tuple[object, ...]) -> bool:
        if not b_args:
            return bool(a_args) or True if not a_args else True
        if a_args and len(a_args) == 2 and len(b_args) == 2:
            return _types_compatible(a_args[0], b_args[0]) and _types_compatible(
                a_args[1], b_args[1]
            )
        return True

    def _compare_single_param_container(
        ob: object, a_args: tuple[object, ...], b_args: tuple[object, ...]
    ) -> bool:
        if not b_args:
            return bool(a_args) or _issubclass_safe(a, ob)
        if a_args and len(a_args) == 1 and len(b_args) == 1:
            return _types_compatible(a_args[0], b_args[0])
        return True

    def _compare_generic_aliases(a: object, b: object, origin_a: object, origin_b: object) -> bool:
        # Normalize bare classes to their origin where possible
        oa = origin_a if origin_a is not None else (a if isinstance(a, type) else None)
        ob = origin_b if origin_b is not None else (b if isinstance(b, type) else None)

        # If one side lacks origin, compare present origin against bare type
        if oa is None or ob is None:
            try:
                if oa is not None and ob is None and isinstance(oa, type) and isinstance(b, type):
                    return issubclass(oa, b)
                if ob is not None and oa is None and isinstance(a, type) and isinstance(ob, type):
                    return issubclass(a, ob)
                ta = a if isinstance(a, type) else type(a)
                tb = b if isinstance(b, type) else type(b)
                return issubclass(ta, tb)
            except Exception:
                return False

        # If origins differ, allow subclassing relationship (e.g., MyList vs list)
        if oa is not ob:
            if not _issubclass_safe(oa, ob):
                return False

        # Compare type arguments when available
        args_a, args_b = get_args(a), get_args(b)

        # Tuple
        if ob is tuple:
            if not args_b:
                return bool(args_a) or _issubclass_safe(a, tuple)
            return _compare_tuple(args_a, args_b)

        # Dict
        if ob is dict:
            return _compare_dict(args_a, args_b)

        # Common single-parameter containers
        single_param_containers = (list, set, frozenset)
        if isinstance(ob, type) and ob in single_param_containers:
            return _compare_single_param_container(ob, args_a, args_b)

        # Generic but not special-cased: compare arg lists if both present and same length
        if args_a and args_b and len(args_a) == len(args_b):
            return all(_types_compatible(ta, tb) for ta, tb in zip(args_a, args_b, strict=True))

        # If one side lacks args, consider it compatible (e.g., List[str] vs list)
        return True

    # Explicit handling for None/type(None) to avoid issubclass errors and crashy validation
    if a is None or a is type(None):  # intentional NoneType check
        # typing wildcard always compatible
        if _is_typing_any(b):
            return True
        # Normalize b for union inspection below
        origin_b = get_origin(b)
        if origin_b is Union:
            return any(_types_compatible(type(None), arg) for arg in get_args(b))
        if hasattr(types, "UnionType") and isinstance(b, types.UnionType):
            try:
                return any(_types_compatible(type(None), arg) for arg in b.__args__)
            except Exception:
                return False
        try:
            # Direct match against NoneType
            return issubclass(type(None), b) if isinstance(b, type) else False
        except Exception:
            return False
    # If a is a value, get its type
    if not isinstance(a, type) and get_origin(a) is None:
        a = type(a)
    if not isinstance(b, type) and get_origin(b) is None:
        b = type(b)

    # Trivial compatibilities
    if _is_typing_any(a) or _is_typing_any(b):
        return True

    origin_a, origin_b = get_origin(a), get_origin(b)

    # Handle unions
    if _is_union_type(b):
        return any(_types_compatible(a, arg) for arg in _iter_union_args(b))
    if _is_union_type(a):
        return all(_types_compatible(arg, b) for arg in _iter_union_args(a))

    # Generic alias handling
    if origin_a is not None or origin_b is not None:
        return _compare_generic_aliases(a, b, origin_a, origin_b)

    # Fallback for non-generic classes
    if not isinstance(a, type) or not isinstance(b, type):
        return False
    try:
        return issubclass(a, b)
    except Exception:
        return False
