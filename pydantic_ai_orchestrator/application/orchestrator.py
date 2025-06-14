"""Backward-compatible facade over :class:`PipelineRunner`."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..domain.agent_protocol import AgentProtocol
from ..domain.pipeline_dsl import Step
from ..domain.models import Candidate, PipelineResult, Task
from .pipeline_runner import PipelineRunner


class Orchestrator:
    """Simple wrapper around :class:`PipelineRunner`."""

    def __init__(
        self,
        review_agent: AgentProtocol[str, Any],
        solution_agent: AgentProtocol[str, Any],
        validator_agent: AgentProtocol[dict[str, Any], Any],
        reflection_agent: AgentProtocol[dict[str, Any], Any] | None = None,
        max_iters: Optional[int] = None,
        k_variants: Optional[int] = None,
        reflection_limit: Optional[int] = None,
    ) -> None:
        _ = reflection_agent, max_iters, k_variants, reflection_limit

        pipeline = (
            Step.review(review_agent, max_retries=3)
            >> Step.solution(solution_agent, max_retries=3)
            >> Step.validate(validator_agent, max_retries=3)
        )
        self.runner = PipelineRunner(pipeline)

    async def run_async(self, task: Task) -> Candidate | None:
        result: PipelineResult = await self.runner.run_async(task.prompt)
        if len(result.step_history) < 2:
            return None

        solution_output = result.step_history[1].output
        final_step = result.step_history[-1]
        if not final_step.success:
            return None

        return Candidate(solution=str(solution_output), score=1.0, checklist=None)

    def run_sync(self, task: Task) -> Candidate | None:
        return asyncio.run(self.run_async(task))
