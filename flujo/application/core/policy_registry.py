from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, Type, TypeVar, cast

from ...domain.dsl.step import Step
from ...domain.models import StepOutcome
from .types import ExecutionFrame

TStep = TypeVar("TStep", bound=Step[Any, Any])
PolicyCallable = Callable[[ExecutionFrame[Any]], Awaitable[StepOutcome[Any]]]


class StepPolicy(ABC, Generic[TStep]):
    """Protocol for step execution policies."""

    @property
    @abstractmethod
    def handles_type(self) -> Type[TStep]:
        """The step type this policy handles."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, core: Any, *args: Any, **kwargs: Any) -> StepOutcome[Any]:
        """Execute a step for the given core + parameters (frame or legacy args)."""
        raise NotImplementedError


class CallableStepPolicy(StepPolicy[Step[Any, Any]]):
    """Adapter to wrap frame-callable policies as StepPolicy instances."""

    def __init__(self, handles_type: Type[Step[Any, Any]], func: PolicyCallable) -> None:
        self._handles_type = handles_type
        self._func = func

    @property
    def handles_type(self) -> Type[Step[Any, Any]]:
        return self._handles_type

    async def execute(self, core: Any, *args: Any, **kwargs: Any) -> StepOutcome[Any]:
        if not args:
            raise TypeError("ExecutionFrame is required for callable policies")
        frame = cast(ExecutionFrame[Any], args[0])
        return await self._func(frame)


class PolicyRegistry:
    """Registry that maps `Step` subclasses to their execution policy (callable or StepPolicy)."""

    def __init__(self) -> None:
        self._registry: Dict[Type[Step[Any, Any]], PolicyCallable | StepPolicy[Any]] = {}
        self._fallback_policy: PolicyCallable | StepPolicy[Any] | None = None

        # Preload any globally registered policies from framework registry (best-effort).
        try:
            try:  # pragma: no cover - import side-effect only
                import flujo.framework  # noqa: F401
            except Exception:
                pass
            from ...framework.registry import get_registered_policies

            for step_cls, policy_instance in get_registered_policies().items():
                self._registry[step_cls] = policy_instance
        except Exception:
            # Framework registry may not be initialized yet; ignore.
            pass

    def register(
        self,
        step_type: Type[Step[Any, Any]] | StepPolicy[Any],
        policy: PolicyCallable | StepPolicy[Any] | None = None,
    ) -> None:
        """Register a policy for a `Step` subclass or a `StepPolicy` instance."""
        from flujo.domain.dsl.step import Step as _BaseStep

        if isinstance(step_type, StepPolicy):
            policy_obj = step_type
            step_cls: Type[Step[Any, Any]] = policy_obj.handles_type
            self._registry[step_cls] = policy_obj
            return

        if not isinstance(step_type, type) or not issubclass(step_type, _BaseStep):
            raise TypeError("step_type must be a subclass of Step")
        if policy is None:
            raise TypeError("policy is required when registering by step type")
        self._registry[step_type] = policy

    def register_callable(self, step_type: Type[Step[Any, Any]], policy: PolicyCallable) -> None:
        """Register a frame-callable policy without wrapping."""
        self._registry[step_type] = policy

    def register_fallback(self, policy: PolicyCallable | StepPolicy[Any]) -> None:
        """Register a fallback policy for unhandled step types."""
        self._fallback_policy = policy

    def get(self, step_type: Type[Step[Any, Any]]) -> Optional[PolicyCallable | StepPolicy[Any]]:
        """Return the policy for `step_type` (or nearest ancestor) or fallback."""
        policy = self._registry.get(step_type)
        if policy is not None:
            return policy
        try:
            for base in step_type.__mro__[1:]:
                if base in self._registry:
                    return self._registry[base]
        except Exception:
            pass
        return self._fallback_policy

    def list_registered(self) -> list[Type[Step[Any, Any]]]:
        """List all registered step types."""
        return list(self._registry.keys())

    def has_exact(self, step_type: Type[Step[Any, Any]]) -> bool:
        """Check whether a policy is registered for the exact step type (no MRO walk)."""
        return step_type in self._registry


def create_default_registry(core: Any) -> PolicyRegistry:
    """Factory to build a registry populated with default policy handlers for a core."""
    registry = PolicyRegistry()
    # Local import to avoid circular dependency at module import time
    from .policy_handlers import PolicyHandlers

    PolicyHandlers(core).register_all(registry)
    return registry
