from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from ....domain.dsl.import_step import ImportStep

from ....domain.models import (
    BaseModel as DomainBaseModel,
    Failure,
    Paused,
    PipelineResult,
    StepResult,
    UsageLimits,
)
from ....exceptions import PausedException
from .executor_helpers import make_execution_frame
from ..quota_manager import build_root_quota

if TYPE_CHECKING:
    from ..executor_core import ExecutorCore


ContextT = TypeVar("ContextT", bound=DomainBaseModel)
ContextSetter = Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
StepExecutor = Callable[..., object]
ChunkCallback = Callable[[object], Awaitable[None]]


class StepHandler:
    """Delegated step handlers to keep ExecutorCore wiring-only."""

    def __init__(self, core: "ExecutorCore[ContextT]") -> None:
        self._core = core

    async def parallel_step(
        self,
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
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
        outcome = await self._core.parallel_step_executor.execute(self._core, frame)
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

    async def conditional_step(
        self,
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
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
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
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

    async def import_step(
        self,
        step: ImportStep,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
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
        outcome = await self._core._import_orchestrator.execute(
            core=self._core,
            step=step,
            data=data,
            context=context,
            resources=resources,
            limits=limits,
            context_setter=context_setter,
            frame=frame,
        )
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

    async def dynamic_router_step(
        self,
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
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
        outcome = await self._core.dynamic_router_step_executor.execute(self._core, frame)
        return self._core._unwrap_outcome_to_step_result(outcome, self._core._safe_step_name(step))

    async def hitl_step(
        self,
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
        stream: bool,
        on_chunk: ChunkCallback | None,
        cache_key: str | None,
        fallback_depth: int,
    ) -> StepResult:
        frame = make_execution_frame(
            self._core,
            step,
            data,
            context,
            resources,
            limits,
            context_setter=context_setter,
            stream=stream,
            on_chunk=on_chunk,
            fallback_depth=fallback_depth,
            result=None,
            quota=self._core._get_current_quota()
            if hasattr(self._core, "_get_current_quota")
            else None,
        )
        return await self._core._hitl_orchestrator.execute(core=self._core, frame=frame)

    async def loop_step(
        self,
        loop_step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
        fallback_depth: int = 0,
    ) -> StepResult:
        current_quota = None
        try:
            if hasattr(self._core, "_get_current_quota"):
                current_quota = self._core._get_current_quota()
        except Exception:
            current_quota = None
        if current_quota is None:
            current_quota = build_root_quota(limits)
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
            quota=current_quota,
        )
        original_context_setter = getattr(self._core, "_context_setter", None)
        quota_token = None
        try:
            self._core._context_setter = context_setter
            try:
                quota_mgr = getattr(self._core, "_quota_manager", None)
                set_fn = getattr(quota_mgr, "set_current_quota", None)
                if callable(set_fn):
                    quota_token = set_fn(current_quota)
            except Exception:
                quota_token = None  # Quota manager unavailable; continue without quota tracking.
            outcome = await self._core.loop_step_executor.execute(self._core, frame)
            return self._core._unwrap_outcome_to_step_result(
                outcome, self._core._safe_step_name(loop_step)
            )
        finally:
            try:
                if quota_token is not None and hasattr(quota_token, "old_value"):
                    quota_mgr = getattr(self._core, "_quota_manager", None)
                    set_fn = getattr(quota_mgr, "set_current_quota", None)
                    if callable(set_fn):
                        set_fn(quota_token.old_value)
            except Exception:
                pass
            self._core._context_setter = original_context_setter

    async def pipeline(
        self,
        pipeline: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
    ) -> PipelineResult[DomainBaseModel]:
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
        step: object,
        data: object,
        context: ContextT | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: ContextSetter | None,
        router_step: object,
    ) -> StepResult:
        rs = router_step if router_step is not None else step
        frame = make_execution_frame(
            self._core,
            rs,
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
        outcome = await self._core.dynamic_router_step_executor.execute(self._core, frame)
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
