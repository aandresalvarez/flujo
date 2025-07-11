from __future__ import annotations

from typing import Any, Optional, TypeVar
import hashlib
import pickle  # nosec B403 - Used for fallback serialization of complex objects in cache keys
import json
from pydantic import Field

from flujo.domain.dsl import Step
from flujo.domain.models import BaseModel, PipelineContext
from flujo.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def cached(
        cls,
        wrapped_step: Step[Any, Any],
        cache_backend: Optional[CacheBackend] = None,
    ) -> "CacheStep[Any, Any]":
        """Create a CacheStep that wraps the given step with caching."""
        return cls(
            name=wrapped_step.name,
            wrapped_step=wrapped_step,
            cache_backend=cache_backend or InMemoryCache(),
        )


def _serialize_for_key(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects to cacheable structures for cache keys."""
    if obj is None:
        return None
    # Special handling for PipelineContext: exclude run_id
    if isinstance(obj, PipelineContext):
        d = obj.model_dump(mode="python")
        d.pop("run_id", None)
        return {k: _serialize_for_key(v) for k, v in d.items()}
    # Special handling for Step: serialize agent by class name
    if isinstance(obj, Step):
        d = obj.model_dump(mode="python")
        if "agent" in d and d["agent"] is not None:
            d["agent"] = type(d["agent"]).__name__
        return {k: _serialize_for_key(v) for k, v in d.items()}
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="python")
    if isinstance(obj, dict):
        return {k: _serialize_for_key(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_serialize_for_key(v) for v in obj]
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', '<unknown>')}.{getattr(obj, '__qualname__', repr(obj))}"
        )
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _generate_cache_key(
    step: Step[Any, Any],
    data: Any,
    context: Any | None = None,
    resources: Any | None = None,
) -> Optional[str]:
    """Return a stable cache key for the step definition and input."""
    payload = {
        "step": _serialize_for_key(step),
        "data": _serialize_for_key(data),
        "context": _serialize_for_key(context),
        "resources": _serialize_for_key(resources),
    }
    try:
        serialized = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(serialized).hexdigest()
    except Exception:
        try:
            serialized = pickle.dumps(payload)  # nosec B403 - Fallback serialization for cache keys
            digest = hashlib.sha256(serialized).hexdigest()
        except Exception:
            return None
    return f"{step.name}:{digest}"
