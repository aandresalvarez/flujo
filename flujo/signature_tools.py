import inspect
import weakref
import types
from typing import Any, Callable, NamedTuple, Optional, get_type_hints, get_origin, get_args, Union

from .infra.telemetry import logfire

from .domain.models import BaseModel
from .domain.resources import AppResources
from .exceptions import ConfigurationError


class InjectionSpec(NamedTuple):
    needs_context: bool
    needs_resources: bool
    context_kw: Optional[str]


_analysis_cache_weak: "weakref.WeakKeyDictionary[Callable[..., Any], InjectionSpec]" = (
    weakref.WeakKeyDictionary()
)
_analysis_cache_id: weakref.WeakValueDictionary[int, InjectionSpec] = weakref.WeakValueDictionary()


def _cache_get(func: Callable[..., Any]) -> InjectionSpec | None:
    try:
        return _analysis_cache_weak.get(func)
    except TypeError:
        return _analysis_cache_id.get(id(func))


def _cache_set(func: Callable[..., Any], spec: InjectionSpec) -> None:
    try:
        _analysis_cache_weak[func] = spec
    except TypeError:
        _analysis_cache_id[id(func)] = spec


def analyze_signature(func: Callable[..., Any]) -> InjectionSpec:
    cached = _cache_get(func)
    if cached is not None:
        return cached

    needs_context = False
    needs_resources = False
    context_kw: Optional[str] = None
    try:
        sig = inspect.signature(func)
    except Exception as e:  # pragma: no cover - defensive
        logfire.debug(f"Could not inspect signature for {func!r}: {e}")
        spec = InjectionSpec(False, False, None)
        _cache_set(func, spec)
        return spec

    try:
        hints = get_type_hints(func)
    except Exception as e:  # pragma: no cover - defensive
        logfire.debug(f"Could not resolve type hints for {func!r}: {e}")
        hints = {}

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
                context_kw = p.name
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
    spec = InjectionSpec(needs_context, needs_resources, context_kw)
    _cache_set(func, spec)
    return spec
