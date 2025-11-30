from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Callable

from ...domain.models import Failure, Paused, PipelineResult, StepResult, UsageLimits
from ...exceptions import PausedException

if TYPE_CHECKING:
    from .executor_core import ExecutorCore


class StepHandler:
    """Delegated step handlers to keep ExecutorCore wiring-only."""

    def __init__(self, core: "ExecutorCore[Any]") -> None:
        self._core: "ExecutorCore[Any]" = core

    async def parallel_step(
        self,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[Callable[..., Any]],
    ) -> StepResult:
        outcome = await self._core.parallel_step_executor.execute(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter,
            step,
            step_executor,
        )
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

    async def conditional_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        fallback_depth: int = 0,
    ) -> StepResult:
        return await self._core._conditional_orchestrator.execute(
            core=self._core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
            fallback_depth=fallback_depth,
        )

    async def cache_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        step_executor: Optional[Callable[..., Any]],
    ) -> StepResult:
        outcome = await self._core.cache_step_executor.execute(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter,
            step_executor,
        )
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

    async def dynamic_router_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> StepResult:
        return await self._core._complex_step_router._handle_dynamic_router_step(
            core=self._core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
        )

    async def hitl_step(
        self,
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        stream: bool,
        on_chunk: Optional[Callable[[Any], Any]],
        cache_key: Optional[str],
        fallback_depth: int,
    ) -> StepResult:
        return await self._core._hitl_orchestrator.execute(
            core=self._core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
            stream=stream,
            on_chunk=on_chunk,
            cache_key=cache_key,
            fallback_depth=fallback_depth,
        )

    async def loop_step(
        self,
        loop_step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        fallback_depth: int = 0,
    ) -> StepResult:
        core_any: Any = self._core
        original_context_setter = getattr(core_any, "_context_setter", None)
        try:
            setattr(core_any, "_context_setter", context_setter)
            outcome = await self._core.loop_step_executor.execute(
                self._core,
                loop_step,
                data,
                context,
                resources,
                limits,
                False,
                None,
                None,
                fallback_depth,
            )
            return self._core._unwrap_outcome_to_step_result(
                outcome, self._core._safe_step_name(loop_step)
            )
        finally:
            setattr(core_any, "_context_setter", original_context_setter)

    async def pipeline(
        self,
        pipeline: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    ) -> PipelineResult[Any]:
        return await self._core._pipeline_orchestrator.execute(
            core=self._core,
            pipeline=pipeline,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
        )

    async def dynamic_router_wrapper(
        self,
        step: Any,
        data: Any,
        context: Any,
        resources: Any,
        limits: Any,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
        router_step: Any,
        step_executor: Optional[Callable[..., Any]],
    ) -> StepResult:
        rs = router_step if router_step is not None else step
        outcome = await self._core.dynamic_router_step_executor.execute(
            self._core, rs, data, context, resources, limits, context_setter
        )
        if isinstance(outcome, Paused):
            raise PausedException(outcome.message)
        if isinstance(outcome, Failure):
            return self._core._unwrap_outcome_to_step_result(
                outcome, self._core._safe_step_name(rs)
            )
        if hasattr(outcome, "step_result"):
            sr = getattr(outcome, "step_result", None)
            if isinstance(sr, StepResult):
                return sr
        return StepResult(
            name=self._core._safe_step_name(rs),
            success=False,
            feedback="Unsupported router outcome",
        )
