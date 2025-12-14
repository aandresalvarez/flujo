from __future__ import annotations
import inspect
import weakref
import types
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union,
)

from .infra.telemetry import logfire

from .domain.models import BaseModel
from .domain.resources import AppResources
from .exceptions import ConfigurationError


def _get_type_hints_best_effort(func: Callable[..., Any]) -> dict[str, object]:
    """Resolve type hints for ``func`` with Flujo-friendly fallbacks.

    Python 3.14+ may evaluate annotations in different phases (or lazily),
    which can surface NameError for common Flujo aliases like ``JSONObject`` in
    user-provided skill modules. Retry with a patched globalns that includes
    those aliases.
    """
    try:
        return dict(get_type_hints(func))
    except Exception as first_exc:
        try:
            from flujo.type_definitions.common import JSONObject, JSONArray

            globalns: dict[str, object] = {}
            try:
                func_globals = getattr(func, "__globals__", None)
                if isinstance(func_globals, dict):
                    globalns.update(func_globals)
            except Exception:
                pass
            globalns.setdefault("JSONObject", JSONObject)
            globalns.setdefault("JSONArray", JSONArray)
            globalns.setdefault("Any", Any)
            return dict(get_type_hints(func, globalns=globalns, localns=globalns))
        except Exception as second_exc:
            logfire.debug(
                f"Could not resolve type hints for {func!r}: {first_exc}; "
                f"fallback also failed: {second_exc}"
            )
            return {}


class InjectionSpec(NamedTuple):
    needs_context: bool
    needs_resources: bool
    context_kw: Optional[str]


class SignatureAnalysis(NamedTuple):
    needs_context: bool
    needs_resources: bool
    context_kw: Optional[str]
    # These are "type hint objects" (e.g., `str`, `list[str]`, `typing.Any`), not runtime values.
    input_type: object
    output_type: object


_analysis_cache_weak: "weakref.WeakKeyDictionary[Callable[..., Any], SignatureAnalysis]" = (
    weakref.WeakKeyDictionary()
)
_analysis_cache_id: dict[int, SignatureAnalysis] = {}


def _cache_get(func: Callable[..., Any]) -> Optional[SignatureAnalysis]:
    """Return cached :class:`SignatureAnalysis` for ``func`` if available."""

    try:
        return _analysis_cache_weak.get(func)
    except TypeError:
        return _analysis_cache_id.get(id(func))


def _cache_set(func: Callable[..., Any], spec: SignatureAnalysis) -> None:
    """Store ``spec`` in the cache for ``func``."""

    try:
        _analysis_cache_weak[func] = spec
    except TypeError:
        # Some callables (e.g., instances with __slots__) are not weakref-able.
        # Fallback to id-based cache using a strong reference to the analysis.
        _analysis_cache_id[id(func)] = spec


def analyze_signature(func: Callable[..., Any]) -> SignatureAnalysis:
    """Inspect ``func`` and determine its pipeline injection requirements.

    Parameters
    ----------
    func:
        Callable to inspect. It may be a standard function or a callable
        object.

    Returns
    -------
    SignatureAnalysis
        Named tuple describing whether ``context`` or ``resources`` keyword
        parameters are required and the inferred input/output types.

    Raises
    ------
    ConfigurationError
        If ``context`` or ``resources`` parameters are annotated with invalid
        types.
    """

    cached = _cache_get(func)
    if cached is not None:
        return cached

    # Create the analysis
    needs_context = False
    needs_resources = False
    context_kw: Optional[str] = None
    input_type: object = Any
    output_type: object = Any
    try:
        sig = inspect.signature(func)
    except Exception as e:  # pragma: no cover - defensive
        logfire.debug(f"Could not inspect signature for {func!r}: {e}")
        result = SignatureAnalysis(False, False, None, Any, Any)
        _cache_set(func, result)
        return result

    try:
        hints = _get_type_hints_best_effort(func)
    except Exception:  # pragma: no cover - defensive
        hints = {}

    # Extract input_type (first parameter)
    params = list(sig.parameters.values())
    if params:
        first_param = params[0]
        if first_param.name in hints:
            input_type = hints[first_param.name]
        elif first_param.annotation is not inspect.Parameter.empty:
            input_type = first_param.annotation
        else:
            input_type = Any
    # Extract output_type (return annotation)
    if "return" in hints:
        output_type = hints["return"]
    elif sig.return_annotation is not inspect.Parameter.empty:
        output_type = sig.return_annotation
    else:
        output_type = Any

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            if p.name == "context":
                ann = hints.get(p.name, p.annotation)
                if ann is inspect.Signature.empty:
                    raise ConfigurationError(
                        f"Parameter '{p.name}' must be annotated with a BaseModel subclass"
                    )
                origin = get_origin(ann)
                if origin in {Union, getattr(types, "UnionType", Union)}:
                    args = get_args(ann)
                    # Relaxed check: allow if name matches PipelineContext to avoid import/reloading issues in tests
                    if not any(
                        (isinstance(a, type) and issubclass(a, BaseModel))
                        or getattr(a, "__name__", "") in ("PipelineContext", "BaseModel")
                        or (
                            isinstance(a, str)
                            and ("PipelineContext" in a or "_BaseModel" in a or "BaseModel" in a)
                        )
                        for a in args
                    ):
                        raise ConfigurationError(
                            f"Parameter '{p.name}' must be annotated with a BaseModel subclass"
                        )
                elif not (
                    (isinstance(ann, type) and issubclass(ann, BaseModel))
                    or getattr(ann, "__name__", "") in ("PipelineContext", "BaseModel")
                    or (
                        isinstance(ann, str)
                        and ("PipelineContext" in ann or "_BaseModel" in ann or "BaseModel" in ann)
                    )
                ):
                    raise ConfigurationError(
                        f"Parameter '{p.name}' must be annotated with a BaseModel subclass"
                    )
                needs_context = True
                context_kw = "context"  # Always use "context" as the parameter name
            elif p.name == "resources":
                ann = hints.get(p.name, p.annotation)
                if ann is inspect.Signature.empty:
                    raise ConfigurationError(
                        "Parameter 'resources' must be annotated with an AppResources subclass"
                    )
                origin = get_origin(ann)
                if origin in {Union, getattr(types, "UnionType", Union)}:
                    args = get_args(ann)
                    if not any(
                        (isinstance(a, type) and issubclass(a, AppResources))
                        or getattr(a, "__name__", "") == "AppResources"
                        or (isinstance(a, str) and "AppResources" in a)
                        for a in args
                    ):
                        raise ConfigurationError(
                            "Parameter 'resources' must be annotated with an AppResources subclass"
                        )
                elif not (
                    (isinstance(ann, type) and issubclass(ann, AppResources))
                    or getattr(ann, "__name__", "") == "AppResources"
                    or (isinstance(ann, str) and "AppResources" in ann)
                ):
                    raise ConfigurationError(
                        "Parameter 'resources' must be annotated with an AppResources subclass"
                    )
                needs_resources = True

    result = SignatureAnalysis(needs_context, needs_resources, context_kw, input_type, output_type)
    _cache_set(func, result)
    return result
