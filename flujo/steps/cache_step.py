from __future__ import annotations

from typing import Any, Optional, TypeVar
import hashlib
import pickle
import orjson
from pydantic import Field

from flujo.domain.pipeline_dsl import Step
from flujo.domain.models import BaseModel, PipelineContext
from flujo.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    model_config = {"arbitrary_types_allowed": True}


def _serialize_for_key(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects to cacheable structures."""
    if obj is None:
        return None
    if isinstance(obj, PipelineContext):
        return obj.model_dump(mode="json", exclude={"run_id"})
    if isinstance(obj, BaseModel):
        return {k: _serialize_for_key(v) for k, v in obj.model_dump(mode="python").items()}
    if isinstance(obj, dict):
        return {k: _serialize_for_key(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_serialize_for_key(v) for v in obj]
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', '<unknown>')}.{getattr(obj, '__qualname__', repr(obj))}"
        )
    try:
        orjson.dumps(obj)
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
        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    except Exception:
        try:
            serialized = pickle.dumps(payload)
        except Exception:
            return None
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{step.name}:{digest}"
