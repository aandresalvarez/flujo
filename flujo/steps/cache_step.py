from __future__ import annotations

from typing import Any, Optional, TypeVar, ClassVar
import hashlib
import pickle  # nosec B403 - Used for fallback serialization of complex objects in cache keys
import orjson
from pydantic import Field

from flujo.domain.dsl import Step
from flujo.domain.dsl.step import StepType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flujo.domain.ir import StepIR
from flujo.domain.models import BaseModel, PipelineContext
from flujo.caching import CacheBackend, InMemoryCache

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Step[StepInT, StepOutT]
    cache_backend: CacheBackend = Field(default_factory=InMemoryCache)

    step_type: ClassVar[StepType] = StepType.CACHE

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

    # ------------------------------------------------------------------
    # IR helpers
    # ------------------------------------------------------------------

    def to_model(self) -> "StepIR":
        base = super().to_model()
        base.step_type = self.step_type.value
        base.wrapped_step = self.wrapped_step.to_model()
        base.cache_backend = self.cache_backend
        return base

    @classmethod
    def from_model(cls, model: "StepIR") -> "CacheStep[Any, Any]":
        wrapped = (
            Step.from_model(model.wrapped_step)
            if model.wrapped_step
            else Step.model_validate({"name": model.name})
        )
        from flujo.domain.plugins import plugin_registry

        step = cls.cached(wrapped, cache_backend=model.cache_backend)
        step.config = model.config
        plugins = []
        for p in model.plugins:
            plugin_cls = plugin_registry.get(p.plugin_type)
            if plugin_cls is not None:
                plugins.append((plugin_cls(), p.priority))
        step.plugins = plugins
        step.validators = []
        step.processors = model.processors
        step.persist_feedback_to_context = model.persist_feedback_to_context
        step.persist_validation_results_to = model.persist_validation_results_to
        step.updates_context = model.updates_context
        step.meta = model.meta
        step.step_uid = model.step_uid
        return step


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
            serialized = pickle.dumps(payload)  # nosec B403 - Fallback serialization for cache keys
        except Exception:
            return None
    digest = hashlib.sha256(serialized).hexdigest()
    return f"{step.name}:{digest}"
