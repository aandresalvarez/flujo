"""Step type routing and dispatch logic."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Type, cast, Union

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
            try:
                if core_obj is None:
                    raise TypeError("Executor core is required for policy execution")
                # Help mypy understand core_obj is non-None below
                assert core_obj is not None
                return await policy.execute(core_obj, frame)
            except TypeError:
                if core_obj is None:
                    raise
                # Backward-compatible call paths for policies still expecting expanded args.
                from ...domain.dsl.loop import LoopStep
                from ...domain.dsl.parallel import ParallelStep
                from ...domain.dsl.conditional import ConditionalStep
                from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
                from ...domain.dsl.step import HumanInTheLoopStep
                from ...steps.cache_step import CacheStep
                from ...domain.dsl.import_step import ImportStep

                try:
                    handles = getattr(policy, "handles_type", None)
                except Exception:
                    handles = None
                # Default to simple/agent signature when type is unknown
                if handles is None or handles is Step:
                    return await policy.execute(
                        core_obj,
                        frame.step,
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.stream,
                        frame.on_chunk,
                        getattr(frame, "cache_key", None),
                        getattr(frame, "_fallback_depth", 0),
                    )
                if handles is LoopStep or (
                    isinstance(handles, type) and issubclass(handles, LoopStep)
                ):
                    cache_key = (
                        core_obj._cache_key(frame)
                        if getattr(core_obj, "_enable_cache", False)
                        else getattr(frame, "cache_key", None)
                    )
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
                        getattr(frame, "_fallback_depth", 0),
                    )
                if handles is ParallelStep or (
                    isinstance(handles, type) and issubclass(handles, ParallelStep)
                ):
                    return await policy.execute(
                        core_obj,
                        frame.step,
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                        cast(ParallelStep[Any], frame.step),
                        None,
                    )
                if handles is ConditionalStep or (
                    isinstance(handles, type) and issubclass(handles, ConditionalStep)
                ):
                    return await policy.execute(
                        core_obj,
                        frame.step,
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                        getattr(frame, "_fallback_depth", 0),
                    )
                if handles is DynamicParallelRouterStep or (
                    isinstance(handles, type) and issubclass(handles, DynamicParallelRouterStep)
                ):
                    return await policy.execute(
                        core_obj,
                        frame.step,
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                        frame.step,
                    )
                if handles is HumanInTheLoopStep or (
                    isinstance(handles, type) and issubclass(handles, HumanInTheLoopStep)
                ):
                    return await policy.execute(
                        core_obj,
                        cast(HumanInTheLoopStep, frame.step),
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                    )
                if handles is CacheStep or (
                    isinstance(handles, type) and issubclass(handles, CacheStep)
                ):
                    return await policy.execute(
                        core_obj,
                        cast(CacheStep[Any, Any], frame.step),
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                        None,
                        frame.step,
                    )
                if handles is ImportStep or (
                    isinstance(handles, type) and issubclass(handles, ImportStep)
                ):
                    return await policy.execute(
                        core_obj,
                        cast(ImportStep, frame.step),
                        frame.data,
                        frame.context,
                        frame.resources,
                        frame.limits,
                        frame.context_setter,
                    )
                # Fallback: try simple/agent signature
                return await policy.execute(
                    core_obj,
                    frame.step,
                    frame.data,
                    frame.context,
                    frame.resources,
                    frame.limits,
                    frame.stream,
                    frame.on_chunk,
                    getattr(frame, "cache_key", None),
                    getattr(frame, "_fallback_depth", 0),
                )
        return await policy(frame)

    @property
    def registry(self) -> PolicyRegistry:
        """Expose underlying registry (compatibility)."""
        return self._registry
