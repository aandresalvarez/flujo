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


class InjectionSpec(NamedTuple):
    needs_context: bool
    needs_resources: bool
    context_kw: Optional[str]


class SignatureAnalysis(NamedTuple):
    needs_context: bool
    needs_resources: bool
    context_kw: Optional[str]
    input_type: Any
    output_type: Any


_analysis_cache_weak: "weakref.WeakKeyDictionary[Callable[..., Any], SignatureAnalysis]" = (
    weakref.WeakKeyDictionary()
)
_analysis_cache_id: weakref.WeakValueDictionary[int, SignatureAnalysis] = (
    weakref.WeakValueDictionary()
)


def _cache_get(func: Callable[..., Any]) -> SignatureAnalysis | None:
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
    input_type = Any
    output_type = Any
    try:
        sig = inspect.signature(func)
    except Exception as e:  # pragma: no cover - defensive
        logfire.debug(f"Could not inspect signature for {func!r}: {e}")
        result = SignatureAnalysis(False, False, None, Any, Any)
        _cache_set(func, result)
        return result

    try:
        hints = get_type_hints(func)
    except Exception as e:  # pragma: no cover - defensive
        logfire.debug(f"Could not resolve type hints for {func!r}: {e}")
        hints = {}

    # Extract input_type (first parameter)
    params = list(sig.parameters.values())
    if params:
        first_param = params[0]
        input_type = hints.get(first_param.name, first_param.annotation)
        if input_type is inspect.Signature.empty:
            input_type = Any
    # Extract output_type (return annotation)
    output_type = hints.get("return", sig.return_annotation)
    if output_type is inspect.Signature.empty:
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
                    if not any(isinstance(a, type) and issubclass(a, BaseModel) for a in args):
                        raise ConfigurationError(
                            f"Parameter '{p.name}' must be annotated with a BaseModel subclass"
                        )
                elif not (isinstance(ann, type) and issubclass(ann, BaseModel)):
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
                    if not any(isinstance(a, type) and issubclass(a, AppResources) for a in args):
                        raise ConfigurationError(
                            "Parameter 'resources' must be annotated with an AppResources subclass"
                        )
                elif not (isinstance(ann, type) and issubclass(ann, AppResources)):
                    raise ConfigurationError(
                        "Parameter 'resources' must be annotated with an AppResources subclass"
                    )
                needs_resources = True

    result = SignatureAnalysis(needs_context, needs_resources, context_kw, input_type, output_type)
    _cache_set(func, result)
    return result
