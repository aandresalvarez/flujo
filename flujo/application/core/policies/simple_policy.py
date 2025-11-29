from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, cast, Type

from flujo.domain.models import Paused, StepOutcome, StepResult, UsageLimits
from flujo.exceptions import PausedException
from flujo.infra import telemetry
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from ....domain.dsl.step import Step

# Backward compatibility: alias kept for consumers/tests expecting this symbol
SimpleStepExecutorOutcomes = StepOutcome[StepResult]


class SimpleStepExecutor(Protocol):
    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]: ...


class DefaultSimpleStepExecutor(StepPolicy[Step[Any, Any]]):
    @property
    def handles_type(self) -> Type[Step[Any, Any]]:
        return Step

    async def execute(
        self,
        core: Any,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]],
        cache_key: Optional[str],
        _fallback_depth: int = 0,
    ) -> StepOutcome[StepResult]:
        # Allow ExecutionDispatcher to pass an ExecutionFrame directly
        if isinstance(step, ExecutionFrame):
            frame = step
            step = frame.step
            data = frame.data
            context = frame.context
            resources = frame.resources
            limits = frame.limits
            stream = frame.stream
            on_chunk = frame.on_chunk
            cache_key = cache_key or (
                core._cache_key(frame) if getattr(core, "_enable_cache", False) else None
            )
            try:
                _fallback_depth = int(getattr(frame, "_fallback_depth", _fallback_depth) or 0)
            except Exception:
                _fallback_depth = _fallback_depth

        telemetry.logfire.debug(
            f"[Policy] SimpleStep: delegating to core orchestration for '{getattr(step, 'name', '<unnamed>')}'"
        )
        try:
            outcome = await core._agent_handler.execute(
                step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                _fallback_depth,
            )
            # Cache successful outcomes here when called directly via policy
            try:
                from flujo.domain.models import Success as _Success

                if (
                    isinstance(outcome, _Success)
                    and cache_key
                    and getattr(core, "_enable_cache", False)
                ):
                    await core._cache_backend.put(cache_key, outcome.step_result, ttl_s=3600)
            except Exception:
                pass
            return cast(StepOutcome[StepResult], outcome)
        except PausedException as e:
            # Surface as Paused outcome to maintain control-flow semantics
            return Paused(message=getattr(e, "message", ""))


async def _execute_simple_step_policy_impl(
    core: Any,
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    stream: bool,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]],
    cache_key: Optional[str],
    _fallback_depth: int,
) -> StepOutcome[StepResult]:
    """Deprecated: orchestration now handled in ExecutorCore; keep legacy entry-point thin."""
    try:
        return cast(
            StepOutcome[StepResult],
            await core._execute_agent_with_orchestration(
                step,
                data,
                context,
                resources,
                limits,
                stream,
                on_chunk,
                cache_key,
                _fallback_depth,
            ),
        )
    except PausedException as e:
        return Paused(message=getattr(e, "message", ""))
