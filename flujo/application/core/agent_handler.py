from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Callable

from ...domain.models import StepOutcome, StepResult, UsageLimits

if TYPE_CHECKING:
    from .executor_core import ExecutorCore


class AgentHandler:
    """Thin wrapper for agent orchestration dispatch."""

    def __init__(self, core: "ExecutorCore[Any]") -> None:
        self._core: "ExecutorCore[Any]" = core

    async def execute(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Any]],
        cache_key: Optional[str],
        fallback_depth: int,
    ) -> StepOutcome[StepResult]:
        return await self._core._agent_orchestrator.execute(
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
