"""Complex step routing extracted from ExecutorCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from ...domain.dsl.conditional import ConditionalStep
from ...domain.dsl.dynamic_router import DynamicParallelRouterStep
from ...domain.dsl.loop import LoopStep
from ...domain.dsl.step import HumanInTheLoopStep
from ...domain.models import PipelineResult, StepResult, UsageLimits

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .executor_core import ExecutorCore


class ComplexStepRouter:
    """Routes complex steps to their handlers."""

    async def route(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        stream: bool = False,
        on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
        fallback_depth: int = 0,
    ) -> StepResult:
        if isinstance(step, LoopStep):
            return await core._handle_loop_step(
                step, data, context, resources, limits, context_setter, fallback_depth
            )
        if isinstance(step, ConditionalStep):
            return await core._handle_conditional_step(
                step, data, context, resources, limits, context_setter, fallback_depth
            )
        if isinstance(step, DynamicParallelRouterStep):
            return await self._handle_dynamic_router_step(
                core=core,
                step=step,
                data=data,
                context=context,
                resources=resources,
                limits=limits,
                context_setter=context_setter,
            )
        if isinstance(step, HumanInTheLoopStep):
            return await core._handle_hitl_step(
                step, data, context, resources, limits, context_setter
            )
        outcome = await core.execute(
            step,
            data,
            context=context,
            resources=resources,
            limits=limits,
            stream=stream,
            on_chunk=on_chunk,
            context_setter=context_setter,
            _fallback_depth=fallback_depth,
        )
        return core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(step))

    async def _handle_dynamic_router_step(
        self,
        *,
        core: "ExecutorCore[Any]",
        step: Any,
        data: Any,
        context: Optional[Any],
        resources: Optional[Any],
        limits: Optional[UsageLimits],
        context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    ) -> StepResult:
        outcome = await core.dynamic_router_step_executor.execute(
            core, step, data, context, resources, limits, context_setter
        )
        return core._unwrap_outcome_to_step_result(outcome, core._safe_step_name(step))
