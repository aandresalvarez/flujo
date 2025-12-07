from __future__ import annotations

from typing import Any, Protocol, cast, Type

from flujo.domain.models import Paused, StepOutcome, StepResult
from flujo.exceptions import PausedException
from flujo.infra import telemetry
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from ....domain.dsl.step import Step

# Backward compatibility: alias kept for consumers/tests expecting this symbol
SimpleStepExecutorOutcomes = StepOutcome[StepResult]


class SimpleStepExecutor(Protocol):
    async def execute(self, core: Any, frame: ExecutionFrame[Any]) -> StepOutcome[StepResult]: ...


class DefaultSimpleStepExecutor(StepPolicy[Step[Any, Any]]):
    @property
    def handles_type(self) -> Type[Step[Any, Any]]:
        return Step

    async def execute(self, core: Any, frame: ExecutionFrame[Any]) -> StepOutcome[StepResult]:
        """Frame-based execution entrypoint for simple steps."""
        step = frame.step
        data = frame.data
        context = frame.context
        resources = frame.resources
        limits = frame.limits
        stream = frame.stream
        on_chunk = frame.on_chunk
        cache_key = None
        if getattr(core, "_enable_cache", False):
            try:
                cache_key = core._cache_key(frame)
            except Exception:
                cache_key = None
        try:
            fallback_depth = int(getattr(frame, "_fallback_depth", 0) or 0)
        except Exception:
            fallback_depth = 0

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
                fallback_depth,
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
