from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Callable

from ...domain.models import Failure, Paused, PipelineResult, StepResult, UsageLimits
from ...exceptions import PausedException
from .executor_helpers import make_execution_frame

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
        frame = make_execution_frame(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter=context_setter,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            result=None,
            quota=self._core._get_current_quota()
            if hasattr(self._core, "_get_current_quota")
            else None,
        )
        setattr(frame, "step_executor", step_executor)
        outcome = await self._core.parallel_step_executor.execute(self._core, frame)
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
        frame = make_execution_frame(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter=context_setter,
            stream=False,
            on_chunk=None,
            fallback_depth=fallback_depth,
            result=None,
            quota=self._core._get_current_quota()
            if hasattr(self._core, "_get_current_quota")
            else None,
        )
        outcome = await self._core._conditional_orchestrator.execute(core=self._core, frame=frame)
        return self._core._unwrap_outcome_to_step_result(
            outcome,
            self._core._safe_step_name(step),
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
        frame = make_execution_frame(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter=context_setter,
            stream=False,
            on_chunk=None,
            fallback_depth=0,
            result=None,
            quota=self._core._get_current_quota()
            if hasattr(self._core, "_get_current_quota")
            else None,
        )
        outcome = await self._core.cache_step_executor.execute(self._core, frame)
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
        outcome = await self._core.dynamic_router_step_executor.execute(
            self._core, step, data, context, resources, limits, context_setter
        )
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

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
        frame = make_execution_frame(
            self._core,
            loop_step,
            data,
            context,
            resources,
            limits,
            context_setter=context_setter,
            stream=False,
            on_chunk=None,
            fallback_depth=fallback_depth,
            result=None,
            quota=(
                self._core._get_current_quota()
                if hasattr(self._core, "_get_current_quota")
                else None
            ),
        )
        original_context_setter = getattr(self._core, "_context_setter", None)
        try:
            try:
                setattr(self._core, "_context_setter", context_setter)
            except Exception:
                pass
            outcome = await self._core.loop_step_executor.execute(self._core, frame)
            return self._core._unwrap_outcome_to_step_result(
                outcome, self._core._safe_step_name(loop_step)
            )
        finally:
            try:
                setattr(self._core, "_context_setter", original_context_setter)
            except Exception:
                pass

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
