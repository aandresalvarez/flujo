from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ....domain.models import BaseModel, StepOutcome, StepResult, UsageLimits

if TYPE_CHECKING:
    from ..executor_core import ExecutorCore


ContextT = TypeVar("ContextT", bound=BaseModel)


class AgentHandler:
    """Thin wrapper for agent orchestration dispatch."""

    def __init__(self, core: "ExecutorCore[ContextT]") -> None:
        self._core: "ExecutorCore[ContextT]" = core

    async def execute(
        self,
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        stream: bool,
        on_chunk: Callable[[object], Awaitable[None]] | None,
        cache_key: str | None,
        fallback_depth: int,
    ) -> StepOutcome[StepResult]:
        orchestrator = getattr(self._core, "_agent_orchestrator", None)
        execute_fn = getattr(orchestrator, "execute", None)
        if not callable(execute_fn):
            raise TypeError("ExecutorCore missing _agent_orchestrator.execute")
        outcome = await execute_fn(
            core=self._core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=fallback_depth,
        )
        if not isinstance(outcome, StepOutcome):
            raise TypeError(f"AgentOrchestrator returned unsupported type {type(outcome).__name__}")
        return outcome
