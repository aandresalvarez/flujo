from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, TYPE_CHECKING

from ...domain.dsl.step import HumanInTheLoopStep
from ...domain.models import PipelineResult, StepOutcome, StepResult, UsageLimits

if TYPE_CHECKING:
    from .executor_core import ExecutorCore


async def handle_parallel_step(
    core: "ExecutorCore[Any]",
    step: Any | None = None,
    data: Any | None = None,
    context: Any | None = None,
    resources: Any | None = None,
    limits: Any | None = None,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    *,
    parallel_step: Any | None = None,
    step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
) -> StepResult:
    ps = parallel_step if parallel_step is not None else step
    return await core._step_handler.parallel_step(
        ps, data, context, resources, limits, context_setter, step_executor
    )


async def execute_pipeline(
    core: "ExecutorCore[Any]",
    pipeline: Any,
    data: Any,
    context: Any,
    resources: Any,
    limits: Any,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
) -> PipelineResult[Any]:
    return await core._step_handler.pipeline(
        pipeline, data, context, resources, limits, context_setter
    )


async def execute_pipeline_via_policies(
    core: "ExecutorCore[Any]",
    pipeline: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[Any],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
) -> PipelineResult[Any]:
    return await core._step_handler.pipeline(
        pipeline, data, context, resources, limits, context_setter
    )


async def handle_loop_step(
    core: "ExecutorCore[Any]",
    loop_step: Any,
    data: Any,
    context: Any,
    resources: Any,
    limits: Any,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    _fallback_depth: int = 0,
) -> StepResult:
    return await core._step_handler.loop_step(
        loop_step, data, context, resources, limits, context_setter, _fallback_depth
    )


async def handle_dynamic_router(
    core: "ExecutorCore[Any]",
    step: Any | None = None,
    data: Any | None = None,
    context: Any | None = None,
    resources: Any | None = None,
    limits: Any | None = None,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    router_step: Any | None = None,
    step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
) -> StepResult:
    rs = router_step if router_step is not None else step
    return await core._step_handler.dynamic_router_wrapper(
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
        rs,
        step_executor,
    )


async def handle_hitl_step(
    core: "ExecutorCore[Any]",
    step: HumanInTheLoopStep,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    stream: bool = False,
    on_chunk: Optional[Callable[[Any], Awaitable[None]]] = None,
    cache_key: Optional[str] = None,
    _fallback_depth: int = 0,
    **kwargs: Any,
) -> StepResult:
    return await core._step_handler.hitl_step(
        step,
        data,
        context,
        resources,
        limits,
        context_setter,
        stream,
        on_chunk,
        cache_key,
        _fallback_depth,
    )


async def execute_loop(
    core: "ExecutorCore[Any]",
    loop_step: Any,
    data: Any,
    context: Any,
    resources: Any,
    limits: Any,
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    _fallback_depth: int = 0,
) -> StepResult:
    return await core._loop_orchestrator.execute(
        core=core,
        loop_step=loop_step,
        data=data,
        context=context,
        resources=resources,
        limits=limits,
        context_setter=context_setter,
        fallback_depth=_fallback_depth,
    )


async def handle_cache_step(
    core: "ExecutorCore[Any]",
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    *,
    step_executor: Optional[Callable[..., Awaitable[StepResult]]] = None,
    **kwargs: Any,
) -> StepResult:
    return await core._step_handler.cache_step(
        step, data, context, resources, limits, context_setter, step_executor
    )


async def handle_conditional_step(
    core: "ExecutorCore[Any]",
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]],
    _fallback_depth: int = 0,
    **kwargs: Any,
) -> StepResult:
    return await core._step_handler.conditional_step(
        step, data, context, resources, limits, context_setter, _fallback_depth
    )


async def handle_dynamic_router_step(
    core: "ExecutorCore[Any]",
    step: Any,
    data: Any,
    context: Optional[Any],
    resources: Optional[Any],
    limits: Optional[UsageLimits],
    context_setter: Optional[Callable[[PipelineResult[Any], Optional[Any]], None]] = None,
    **kwargs: Any,
) -> StepResult:
    return await core._step_handler.dynamic_router_step(
        step, data, context, resources, limits, context_setter
    )


def default_set_final_context(result: PipelineResult[Any], context: Optional[Any]) -> None:
    """Legacy no-op hook for context propagation."""
    return None


async def execute_agent_with_orchestration(
    core: "ExecutorCore[Any]",
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
    """Compatibility wrapper: agent orchestration now lives in AgentHandler."""
    return await core._agent_handler.execute(
        step=step,
        data=data,
        context=context,
        resources=resources,
        limits=limits,
        stream=stream,
        on_chunk=on_chunk,
        cache_key=cache_key,
        fallback_depth=_fallback_depth,
    )
