"""Opinionated default workflow built on top of :class:`Flujo`."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..infra.agents import AsyncAgentProtocol

from ..domain.pipeline_dsl import Step
from ..domain.models import Candidate, PipelineResult, Task
from ..application.flujo_engine import Flujo


class Default:
    """Pre-configured workflow using the :class:`Flujo` engine."""

    def __init__(
        self,
        review_agent: "AsyncAgentProtocol[Any, Any]",
        solution_agent: "AsyncAgentProtocol[Any, Any]",
        validator_agent: "AsyncAgentProtocol[Any, Any]",
        reflection_agent: "AsyncAgentProtocol[Any, Any]" | None = None,
        max_iters: Optional[int] = None,
        k_variants: Optional[int] = None,
        reflection_limit: Optional[int] = None,
    ) -> None:
        _ = reflection_agent, max_iters, k_variants, reflection_limit

        pipeline = (
            Step.review(cast(Any, review_agent), max_retries=3)
            >> Step.solution(cast(Any, solution_agent), max_retries=3)
            >> Step.validate_step(cast(Any, validator_agent), max_retries=3)
        )
        self.flujo_engine = Flujo(pipeline)

    async def run_async(self, task: Task) -> Candidate | None:
        result: PipelineResult = await self.flujo_engine.run_async(task.prompt)
        if len(result.step_history) < 2:
            return None

        solution_output = result.step_history[1].output
        final_step = result.step_history[-1]
        if not final_step.success:
            return None

        return Candidate(solution=str(solution_output), score=1.0, checklist=None)

    def run_sync(self, task: Task) -> Candidate | None:
        return asyncio.run(self.run_async(task))
