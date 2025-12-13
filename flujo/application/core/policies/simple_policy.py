from __future__ import annotations

from typing import Protocol, Type

from flujo.domain.models import BaseModel, StepOutcome, StepResult
from flujo.exceptions import PausedException
from flujo.infra import telemetry
from ..policy_registry import StepPolicy
from ..types import ExecutionFrame
from ..type_guards import normalize_outcome
from ....domain.dsl.step import Step

# Backward compatibility: alias kept for consumers/tests expecting this symbol
SimpleStepExecutorOutcomes = StepOutcome[StepResult]


class SimpleStepExecutor(Protocol):
    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]: ...


class DefaultSimpleStepExecutor(StepPolicy[Step[object, object]]):
    @property
    def handles_type(self) -> Type[Step[object, object]]:
        return Step

    async def execute(
        self, core: object, frame: ExecutionFrame[BaseModel]
    ) -> StepOutcome[StepResult]:
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
                cache_key_fn = getattr(core, "_cache_key", None)
                if callable(cache_key_fn):
                    cache_key = cache_key_fn(frame)
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
            agent_handler = getattr(core, "_agent_handler", None)
            execute_fn = getattr(agent_handler, "execute", None)
            if not callable(execute_fn):
                raise TypeError("ExecutorCore missing _agent_handler.execute")
            outcome = await execute_fn(
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
                    ttl_s = getattr(core, "_cache_ttl_s", 3600)
                    backend = getattr(core, "_cache_backend", None)
                    put_fn = getattr(backend, "put", None)
                    if callable(put_fn):
                        await put_fn(cache_key, outcome.step_result, ttl_s=ttl_s)
            except Exception:
                pass
            return normalize_outcome(outcome, step_name=getattr(step, "name", "<unnamed>"))
        except PausedException as e:
            # Control-flow exception: must propagate to the runner (do not coerce into data).
            raise e
