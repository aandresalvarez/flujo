"""Step type routing and dispatch logic."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Optional, Type, Union

from ...domain.dsl.step import Step
from ...domain.models import Failure, StepOutcome, StepResult
from .types import ExecutionFrame
from .policy_registry import PolicyRegistry, StepPolicy


PolicyCallable = Callable[[ExecutionFrame[Any]], Awaitable[StepOutcome[Any]]]
RegisteredPolicy = Union[PolicyCallable, StepPolicy[Any]]


class ExecutionDispatcher:
    """Routes step execution to the appropriate policy handler."""

    def __init__(
        self, registry: Optional[PolicyRegistry] = None, *, core: Optional[Any] = None
    ) -> None:
        self._registry: PolicyRegistry = registry or PolicyRegistry()
        self._core = core

    def register(self, step_type: Type[Step[Any, Any]], policy: PolicyCallable) -> None:
        """Register a policy; thin wrapper around PolicyRegistry."""
        self._registry.register(step_type, policy)

    def get_policy(self, step: Step[Any, Any]) -> Optional[RegisteredPolicy]:
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
        if isinstance(policy, StepPolicy):
            core_obj: Any | None = (
                self._core if self._core is not None else getattr(frame, "core", None)
            )
            if core_obj is None:
                raise TypeError("Executor core is required for policy execution")
            if self._expects_frame(policy):
                return await policy.execute(core_obj, frame)
            return await self._call_legacy_policy(policy, core_obj, frame)
        return await policy(frame)

    def _expects_frame(self, policy: StepPolicy[Any]) -> bool:
        """Detect whether a policy prefers the new frame-based signature."""
        try:
            sig = inspect.signature(policy.execute)
            params = list(sig.parameters.values())
            # Bound method: first param is usually `core`
            for param in params[1:]:
                if param.name == "frame":
                    return True
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    # Policies using *args typically expect a frame as the first arg
                    return True
                ann = param.annotation
                if ann is ExecutionFrame or getattr(ann, "__name__", "") == "ExecutionFrame":
                    return True
        except Exception:
            # Default to legacy behavior if inspection fails
            return False
        return False

    async def _call_legacy_policy(
        self, policy: StepPolicy[Any], core_obj: Any, frame: ExecutionFrame[Any]
    ) -> StepOutcome[Any]:
        """Adapt ExecutionFrame into legacy positional arguments for old policies."""
        cache_key = None
        try:
            if getattr(core_obj, "_enable_cache", False):
                cache_key = core_obj._cache_key(frame)  # type: ignore[attr-defined]
        except Exception:
            cache_key = None
        fallback_depth = 0
        try:
            fallback_depth = int(getattr(frame, "_fallback_depth", 0) or 0)
        except Exception:
            fallback_depth = 0

        return await policy.execute(
            core_obj,
            frame.step,
            frame.data,
            frame.context,
            frame.resources,
            frame.limits,
            frame.stream,
            frame.on_chunk,
            cache_key,
            fallback_depth,
        )

    @property
    def registry(self) -> PolicyRegistry:
        """Expose underlying registry (compatibility)."""
        return self._registry
