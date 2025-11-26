"""Step type routing and dispatch logic."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Type

from ...domain.dsl.step import Step
from ...domain.models import Failure, StepOutcome, StepResult
from .types import ExecutionFrame
from .policy_primitives import PolicyRegistry


PolicyCallable = Callable[[ExecutionFrame[Any]], Awaitable[StepOutcome[Any]]]


class ExecutionDispatcher:
    """Routes step execution to the appropriate policy handler."""

    def __init__(self, registry: Optional[PolicyRegistry] = None) -> None:
        self._registry: PolicyRegistry = registry or PolicyRegistry()

    def register(self, step_type: Type[Step[Any, Any]], policy: PolicyCallable) -> None:
        """Register a policy; thin wrapper around PolicyRegistry."""
        self._registry.register(step_type, policy)

    def get_policy(self, step: Step[Any, Any]) -> Optional[PolicyCallable]:
        """Return the policy callable for a step instance, if any."""
        policy = self._registry.get(type(step))
        if policy is None:
            policy = self._registry.get(Step)
        return policy

    async def dispatch(self, frame: ExecutionFrame[Any]) -> StepOutcome[Any]:
        """Dispatch execution to the appropriate policy or return a Failure."""
        step = frame.step
        policy = self.get_policy(step)
        if policy is None:
            return Failure(
                error=TypeError(f"No policy registered for step type: {type(step).__name__}"),
                feedback=f"Unhandled step type: {type(step).__name__}",
                step_result=StepResult(name=step.name, success=False),
            )
        return await policy(frame)

    @property
    def registry(self) -> PolicyRegistry:
        """Expose underlying registry (compatibility)."""
        return self._registry
