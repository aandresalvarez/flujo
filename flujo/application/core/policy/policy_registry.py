from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Generic, Protocol, TypeVar, TYPE_CHECKING

from ....domain.models import BaseModel, StepOutcome, StepResult
from ..types import ExecutionFrame, TContext

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


class StepType(Protocol):
    """Minimal step contract for policy registration."""

    name: str


TStep_co = TypeVar("TStep_co", bound=StepType, covariant=True)
PolicyCallable = Callable[[ExecutionFrame[BaseModel]], Awaitable[StepOutcome[StepResult]]]


class StepPolicy(ABC, Generic[TStep_co]):
    """Protocol for step execution policies."""

    @property
    @abstractmethod
    def handles_type(self) -> type[TStep_co]:
        """The step type this policy handles."""
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]:
        """Execute a step for the given core + execution frame."""
        raise NotImplementedError


class CallableStepPolicy(StepPolicy[StepType]):
    """Adapter to wrap frame-callable policies as StepPolicy instances."""

    def __init__(self, handles_type: type[StepType], func: PolicyCallable) -> None:
        self._handles_type = handles_type
        self._func = func

    @property
    def handles_type(self) -> type[StepType]:
        return self._handles_type

    async def execute(
        self, _core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]:
        return await self._func(frame)


class PolicyRegistry:
    """Registry that maps `Step` subclasses to their execution policy (callable or StepPolicy)."""

    def __init__(self) -> None:
        self._registry: dict[type[StepType], PolicyCallable | StepPolicy[StepType]] = {}
        self._fallback_policy: PolicyCallable | StepPolicy[StepType] | None = None
        self._lookup_cache: dict[type[StepType], PolicyCallable | StepPolicy[StepType] | None] = {}

        # Preload any globally registered policies from framework registry (best-effort).
        try:
            try:  # pragma: no cover - import side-effect only
                import flujo.framework  # noqa: F401
            except Exception:
                pass
            from ....framework.registry import get_registered_policies

            for step_cls, policy_instance in get_registered_policies().items():
                self._registry[step_cls] = policy_instance
            self._invalidate_cache()
        except Exception:
            # Framework registry may not be initialized yet; ignore.
            pass

    def _invalidate_cache(self) -> None:
        self._lookup_cache.clear()

    def register(
        self,
        step_type: type[StepType] | StepPolicy[StepType],
        policy: PolicyCallable | StepPolicy[StepType] | None = None,
    ) -> None:
        """Register a policy for a `Step` subclass or a `StepPolicy` instance."""
        if isinstance(step_type, StepPolicy):
            policy_obj = step_type
            step_cls: type[StepType] = policy_obj.handles_type
            self._registry[step_cls] = policy_obj
            self._invalidate_cache()
            return

        if not isinstance(step_type, type):
            raise TypeError("step_type must be a type")
        # Runtime guard to reject non-Step registrations (maintain policy safety)
        try:
            from ....domain.dsl.step import Step as DSLStep
        except (ImportError, ModuleNotFoundError):
            if policy is None:
                raise
        else:
            if not issubclass(step_type, DSLStep):
                raise TypeError("step_type must be a subclass of Step")
        if policy is None:
            raise TypeError("policy is required when registering by step type")
        self._registry[step_type] = policy
        self._invalidate_cache()

    def register_callable(self, step_type: type[StepType], policy: PolicyCallable) -> None:
        """Register a frame-callable policy without wrapping."""
        self._registry[step_type] = policy
        self._invalidate_cache()

    def register_fallback(self, policy: PolicyCallable | StepPolicy[StepType]) -> None:
        """Register a fallback policy for unhandled step types."""
        self._fallback_policy = policy
        self._invalidate_cache()

    def get(self, step_type: type[StepType]) -> PolicyCallable | StepPolicy[StepType] | None:
        """Return the policy for `step_type` (or nearest ancestor) or fallback."""
        if step_type in self._lookup_cache:
            return self._lookup_cache[step_type]

        policy = self._registry.get(step_type)
        if policy is not None:
            self._lookup_cache[step_type] = policy
            return policy
        try:
            for base in step_type.__mro__[1:]:
                if base in self._registry:
                    resolved = self._registry[base]
                    self._lookup_cache[step_type] = resolved
                    return resolved
        except Exception:
            pass
        self._lookup_cache[step_type] = self._fallback_policy
        return self._fallback_policy

    def list_registered(self) -> list[type[StepType]]:
        """List all registered step types."""
        return list(self._registry.keys())

    def has_exact(self, step_type: type[StepType]) -> bool:
        """Check whether a policy is registered for the exact step type (no MRO walk)."""
        return step_type in self._registry


def create_default_registry(core: "ExecutorCore[TContext]") -> PolicyRegistry:
    """Factory to build a registry populated with default policy handlers for a core."""
    registry = PolicyRegistry()
    # Local import to avoid circular dependency at module import time
    from .policy_handlers import PolicyHandlers

    PolicyHandlers(core).register_all(registry)
    return registry
