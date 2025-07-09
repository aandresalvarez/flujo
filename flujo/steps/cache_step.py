from __future__ import annotations

from typing import Any, Optional, TypeVar, TYPE_CHECKING, Set, cast
import hashlib
import pickle  # nosec B403 - Used for fallback serialization of complex objects in cache keys
import orjson
from pydantic import Field

from flujo.domain.dsl import Step
from flujo.domain.models import BaseModel, PipelineContext
from flujo.caching import CacheBackend, InMemoryCache
from flujo.domain.processors import AgentProcessors

if TYPE_CHECKING:
    from flujo.domain.ir import StepIR, CacheStepIR
    from flujo.registry import CallableRegistry

StepInT = TypeVar("StepInT")
StepOutT = TypeVar("StepOutT")


class CacheStep(Step[StepInT, StepOutT]):
    """Wraps another step to cache its successful results."""

    wrapped_step: Any  # Allow any concrete step type, not just the abstract base class
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

    def to_model(
        self,
        callable_registry: Optional["CallableRegistry"] = None,
        visited: Optional[Set[str]] = None,
    ) -> "StepIR":
        """Convert this CacheStep to its IR representation."""
        from flujo.domain.ir import CacheStepIR, StepConfigIR, ProcessorIR, StepType

        # Create base step IR directly since Step doesn't have to_model()
        base_ir: CacheStepIR[Any, Any] = CacheStepIR[Any, Any](
            step_type=StepType.CACHE,
            name=self.name,
            agent=None,  # CacheStep doesn't have an agent
            config=StepConfigIR(
                max_retries=self.config.max_retries,
                timeout_seconds=self.config.timeout_s,
                temperature=self.config.temperature,
            ),
            plugins=[],
            validators=[],
            processors=ProcessorIR(),
            persist_feedback_to_context=self.persist_feedback_to_context,
            persist_validation_results_to=self.persist_validation_results_to,
            updates_context=self.updates_context,
            meta=self.meta,
            step_uid=self.step_uid,
            wrapped_step=self.wrapped_step.to_model(callable_registry),
            cache_backend={"type": self.cache_backend.__class__.__name__},
        )

        return base_ir

    @classmethod
    def _from_cache_ir(
        cls,
        ir_model: "CacheStepIR[Any, Any]",
        agent_registry: Optional[dict[str, Any]] = None,
        callable_registry: Optional["CallableRegistry"] = None,
    ) -> "CacheStep[Any, Any]":
        """Create a CacheStep from its IR representation."""
        from flujo.domain.dsl.step import StepConfig
        from flujo.domain.ir import (
            StepType,
        )

        # Correctly dispatch to the right Step subclass for wrapped_step
        wrapped_ir = ir_model.wrapped_step
        wrapped_step_type = wrapped_ir.step_type

        if wrapped_step_type == StepType.STANDARD:
            from flujo.domain.dsl.step import StandardStep
            from flujo.domain.ir import StandardStepIR

            wrapped_step_standard: StandardStep[Any, Any] = StandardStep._from_standard_ir(
                cast("StandardStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.LOOP:
            from flujo.domain.dsl.loop import LoopStep
            from flujo.domain.ir import LoopStepIR

            wrapped_step_loop: LoopStep[Any] = LoopStep._from_loop_ir(
                cast("LoopStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.MAP:
            from flujo.domain.dsl.loop import MapStep
            from flujo.domain.ir import MapStepIR

            wrapped_step_map: MapStep[Any] = MapStep._from_map_ir(
                cast("MapStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.CONDITIONAL:
            from flujo.domain.dsl.conditional import ConditionalStep
            from flujo.domain.ir import ConditionalStepIR

            wrapped_step_conditional: ConditionalStep[Any] = ConditionalStep._from_conditional_ir(
                cast("ConditionalStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.PARALLEL:
            from flujo.domain.dsl.parallel import ParallelStep
            from flujo.domain.ir import ParallelStepIR

            wrapped_step_parallel: ParallelStep[Any] = ParallelStep._from_parallel_ir(
                cast("ParallelStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.CACHE:
            wrapped_step_cache: CacheStep[Any, Any] = CacheStep._from_cache_ir(
                cast("CacheStepIR[Any, Any]", wrapped_ir), agent_registry, callable_registry
            )
        elif wrapped_step_type == StepType.HUMAN_IN_THE_LOOP:
            from flujo.domain.dsl.step import HumanInTheLoopStep
            from flujo.domain.ir import HumanInTheLoopStepIR

            wrapped_step_hitl: HumanInTheLoopStep = HumanInTheLoopStep._from_hitl_ir(
                cast("HumanInTheLoopStepIR[Any, Any]", wrapped_ir),
                agent_registry,
                callable_registry,
            )
        else:
            raise ValueError(f"Unknown step type: {wrapped_step_type}")

        # Cache backend config (simplified)
        cache_backend = InMemoryCache()  # Always use in-memory for test/demo

        # Create step config
        config = StepConfig(
            max_retries=ir_model.config.max_retries,
            timeout_s=ir_model.config.timeout_seconds,
            temperature=ir_model.config.temperature,
        )

        return cls(
            name=ir_model.name,
            wrapped_step=wrapped_step_standard
            if wrapped_step_type == StepType.STANDARD
            else wrapped_step_loop
            if wrapped_step_type == StepType.LOOP
            else wrapped_step_map
            if wrapped_step_type == StepType.MAP
            else wrapped_step_conditional
            if wrapped_step_type == StepType.CONDITIONAL
            else wrapped_step_parallel
            if wrapped_step_type == StepType.PARALLEL
            else wrapped_step_cache
            if wrapped_step_type == StepType.CACHE
            else wrapped_step_hitl,
            cache_backend=cache_backend,
            config=config,
            plugins=[],
            validators=[],
            processors=AgentProcessors(),
            persist_feedback_to_context=ir_model.persist_feedback_to_context,
            persist_validation_results_to=ir_model.persist_validation_results_to,
            updates_context=ir_model.updates_context,
            meta=ir_model.meta,
            step_uid=ir_model.step_uid,
        )

    async def arun(self, data: StepInT, **kwargs: Any) -> StepOutT:
        """Run this cache step's wrapped step directly for testing purposes."""
        # Execute the wrapped step directly
        result: Any = await self.wrapped_step.arun(data, **kwargs)
        return cast(StepOutT, result)


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
