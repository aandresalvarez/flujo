"""Opinionated default workflow built on top of :class:`Flujo`."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, TYPE_CHECKING, cast

from flujo.domain.models import PipelineContext

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..infra.agents import AsyncAgentProtocol

from ..domain.pipeline_dsl import Step
from ..domain.models import Candidate, PipelineResult, Task, Checklist
from ..domain.scoring import ratio_score
from ..application.flujo_engine import Flujo
from ..testing.utils import gather_result


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
        _ = max_iters, k_variants, reflection_limit

        async def _invoke(target: Any, data: Any) -> Any:
            if hasattr(target, "run") and callable(getattr(target, "run")):
                return await target.run(data)
            return await target(data)

        class ReviewWrapper:
            async def run(self, prompt: str, *, pipeline_context: PipelineContext) -> str:
                result = await _invoke(review_agent, prompt)
                checklist = cast(Checklist, getattr(result, "output", result))
                pipeline_context.scratchpad["checklist"] = checklist
                return prompt

        class SolutionWrapper:
            async def run(self, prompt: str, *, pipeline_context: PipelineContext) -> str:
                result = await _invoke(solution_agent, prompt)
                solution = cast(str, getattr(result, "output", result))
                pipeline_context.scratchpad["solution"] = solution
                return solution

        class ValidatorWrapper:
            async def run(self, _data: str, *, pipeline_context: PipelineContext) -> Checklist:
                payload = {
                    "solution": pipeline_context.scratchpad.get("solution"),
                    "checklist": pipeline_context.scratchpad.get("checklist"),
                }
                result = await _invoke(validator_agent, payload)
                validated = cast(Checklist, getattr(result, "output", result))
                pipeline_context.scratchpad["checklist"] = validated
                return validated

        pipeline = (
            Step.review(ReviewWrapper(), max_retries=3)
            >> Step.solution(SolutionWrapper(), max_retries=3)
            >> Step.validate_step(ValidatorWrapper(), max_retries=3)
        )

        if reflection_agent is not None:
            async def reflection_step(_: Any, *, pipeline_context: PipelineContext) -> str:
                payload = {
                    "solution": pipeline_context.scratchpad.get("solution"),
                    "checklist": pipeline_context.scratchpad.get("checklist"),
                }
                result = await _invoke(reflection_agent, payload)
                reflection = cast(str, getattr(result, "output", result))
                pipeline_context.scratchpad["reflection"] = reflection
                return reflection

            pipeline = pipeline >> Step.from_callable(reflection_step, name="reflection", max_retries=3)

        self.flujo_engine = Flujo(pipeline, context_model=PipelineContext)

    async def run_async(self, task: Task) -> Candidate | None:
        result: PipelineResult = await gather_result(
            self.flujo_engine,
            task.prompt,
            initial_context_data={"initial_prompt": task.prompt},
        )
        ctx = cast(PipelineContext, result.final_pipeline_context)
        solution = cast(Optional[str], ctx.scratchpad.get("solution"))
        checklist = cast(Optional[Checklist], ctx.scratchpad.get("checklist"))
        if solution is None or checklist is None:
            return None

        score = ratio_score(checklist)
        return Candidate(solution=solution, score=score, checklist=checklist)

    def run_sync(self, task: Task) -> Candidate | None:
        return asyncio.run(self.run_async(task))
