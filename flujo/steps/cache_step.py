from __future__ import annotations

from typing import Any, Optional, TypeVar, List
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
        try:
            d = obj.model_dump(mode="json")
            d.pop("run_id", None)
            return {k: _serialize_for_key(v) for k, v in d.items()}
        except (ValueError, RecursionError):
            # Handle circular references
            return f"<{type(obj).__name__} circular>"
    # Special handling for Step: serialize agent by class name
    if isinstance(obj, Step):
        try:
            d = obj.model_dump(mode="json")
            if "agent" in d and d["agent"] is not None:
                # Get the original agent object, not the serialized value
                original_agent = getattr(obj, "agent", None)
                if original_agent is not None:
                    d["agent"] = type(original_agent).__name__
            return {k: _serialize_for_key(v) for k, v in d.items()}
        except (ValueError, RecursionError):
            return f"<{type(obj).__name__} circular>"
    # Always check BaseModel before list/tuple/set
    if isinstance(obj, BaseModel):
        try:
            d = obj.model_dump(mode="json")
            if "run_id" in d:
                d.pop("run_id", None)
            # Recursively process all fields, ensuring lists of models are properly serialized
            result_dict = {}
            for k, v in d.items():
                if isinstance(v, (list, tuple, set, frozenset)):
                    result_dict[k] = _serialize_list_for_key(list(v))
                else:
                    result_dict[k] = _serialize_for_key(v)
            return result_dict
        except (ValueError, RecursionError):
            # Handle circular references
            return f"<{type(obj).__name__} circular>"
    if isinstance(obj, dict):
        d = dict(obj)
        if "run_id" in d and "initial_prompt" in d:
            d.pop("run_id", None)
        result: dict[Any, Any] = {}
        for k, v in d.items():
            if hasattr(v, "model_dump"):
                try:
                    v_dict = v.model_dump(mode="json")
                    if "run_id" in v_dict:
                        v_dict.pop("run_id", None)
                    result[k] = {kk: _serialize_for_key(vv) for kk, vv in v_dict.items()}
                except (ValueError, RecursionError):
                    result[k] = f"<{type(v).__name__} circular>"
            elif isinstance(v, (list, tuple, set, frozenset)):
                result[k] = _serialize_list_for_key(list(v))
            elif isinstance(v, dict):
                result[k] = {kk: _serialize_for_key(vv) for kk, vv in v.items()}
            else:
                result[k] = _serialize_for_key(v)
        return result
    if isinstance(obj, (list, tuple, set, frozenset)):
        return _serialize_list_for_key(list(obj))
    if callable(obj):
        return (
            f"{getattr(obj, '__module__', '<unknown>')}.{getattr(obj, '__qualname__', repr(obj))}"
        )
    # Avoid handling strings based on assumptions about their format.
    # Structured serialization should handle `run_id` exclusion where applicable.
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def _serialize_list_for_key(obj_list: List[Any]) -> List[Any]:
    """Helper function to serialize lists, ensuring BaseModel instances are converted to dicts."""
    result_list: List[Any] = []
    for v in obj_list:
        if hasattr(v, "model_dump"):
            d = v.model_dump(mode="json")
            if "run_id" in d:
                d.pop("run_id", None)
            result_list.append({k: _serialize_for_key(val) for k, val in d.items()})
        elif isinstance(v, dict):
            result_list.append({k: _serialize_for_key(val) for k, val in v.items()})
        elif isinstance(v, (list, tuple, set, frozenset)):
            result_list.append(_serialize_list_for_key(list(v)))
        else:
            result_list.append(_serialize_for_key(v))
    return result_list


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
