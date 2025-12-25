"""Import step orchestration extracted from ExecutorCore."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ....domain.dsl.import_step import ImportStep
from ....domain.models import (
    BaseModel as DomainBaseModel,
    PipelineResult,
    StepOutcome,
    StepResult,
    UsageLimits,
)
from ..execution.executor_helpers import make_execution_frame
from ..step_policies import ImportStepExecutor
from ..types import ExecutionFrame, TContext

if TYPE_CHECKING:  # pragma: no cover
    from ..executor_core import ExecutorCore


class ImportOrchestrator:
    """Executes ImportStep via the configured import_step_executor."""

    def __init__(self, executor: ImportStepExecutor | None) -> None:
        self._executor = executor

    async def execute(
        self,
        *,
        core: "ExecutorCore[TContext]",
        step: ImportStep,
        data: object,
        context: DomainBaseModel | None,
        resources: object | None,
        limits: UsageLimits | None,
        context_setter: Callable[[PipelineResult[DomainBaseModel], DomainBaseModel | None], None]
        | None,
        frame: ExecutionFrame[DomainBaseModel] | None = None,
    ) -> StepOutcome[StepResult]:
        if frame is None:
            frame = make_execution_frame(
                core,
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
                quota=core._get_current_quota() if hasattr(core, "_get_current_quota") else None,
            )
        if self._executor is None:
            result = await core._policy_default_step(frame)
            return result
        executor_result: StepOutcome[StepResult] = await self._executor.execute(core, frame)
        return executor_result
