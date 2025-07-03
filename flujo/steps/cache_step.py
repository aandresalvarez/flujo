from __future__ import annotations

from typing import Any, Optional, TypeVar
import hashlib
import pickle
import orjson
from pydantic import Field

from flujo.domain.pipeline_dsl import Step
from flujo.domain.models import BaseModel
from flujo.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    model_config = {"arbitrary_types_allowed": True}


def _generate_cache_key(step_name: str, data: Any) -> Optional[str]:
    """Return a stable cache key for the step input."""
    try:
        if isinstance(data, BaseModel):
            payload = data.model_dump(mode="json")
        else:
            payload = data
        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    except Exception:
        try:
            serialized = pickle.dumps(data)
        except Exception:
            return None
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{step_name}:{digest}"
