"""Opinionated default workflow built on top of :class:`Flujo`."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, cast, TYPE_CHECKING

from flujo.domain.models import BaseModel

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..infra.agents import AsyncAgentProtocol

from ..domain.pipeline_dsl import Step
from ..domain.models import Candidate, PipelineResult, Task, Checklist
from ..domain.scoring import ratio_score
from ..application.flujo_engine import Flujo
from ..testing.utils import gather_result


class Default:
    """Pre-configured workflow using the :class:`Flujo` engine."""

    class Context(BaseModel):
        """Shared state for the Default recipe."""

        initial_prompt: str
        checklist: Checklist | None = None
        solution: str | None = None

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

        step_review = Step.review(self._wrap_review_agent(review_agent), max_retries=3)
        step_solution = Step.solution(self._wrap_solution_agent(solution_agent), max_retries=3)
        step_validate = Step.validate_step(
            self._wrap_validator_agent(validator_agent), max_retries=3
        )

        pipeline = step_review >> step_solution >> step_validate
        self.flujo_engine = Flujo(pipeline, context_model=Default.Context)

    def _wrap_review_agent(self, agent: "AsyncAgentProtocol[Any, Any]") -> Any:
        async def _invoke(target: Any, data: Any, **kwargs: Any) -> Any:
            if hasattr(target, "run") and callable(getattr(target, "run")):
                return await target.run(data, **kwargs)
            return await target(data, **kwargs)

        class _ReviewAgent:
            async def run(self, prompt: str, *, pipeline_context: Default.Context) -> Checklist:
                result = await _invoke(agent, prompt)
                # Unpack the result if it has an 'output' attribute (AgentRunResult)
                unpacked_result = getattr(result, "output", result)
                pipeline_context.checklist = cast(Checklist, unpacked_result)
                return cast(Checklist, unpacked_result)

        return _ReviewAgent()

    def _wrap_solution_agent(self, agent: "AsyncAgentProtocol[Any, Any]") -> Any:
        async def _invoke(target: Any, data: Any, **kwargs: Any) -> Any:
            if hasattr(target, "run") and callable(getattr(target, "run")):
                return await target.run(data, **kwargs)
            return await target(data, **kwargs)

        class _SolutionAgent:
            async def run(self, _data: Any, *, pipeline_context: Default.Context) -> str:
                result = await _invoke(agent, pipeline_context.initial_prompt)
                # Unpack the result if it has an 'output' attribute (AgentRunResult)
                unpacked_result = getattr(result, "output", result)
                pipeline_context.solution = cast(str, unpacked_result)
                return cast(str, unpacked_result)

        return _SolutionAgent()

    def _wrap_validator_agent(self, agent: "AsyncAgentProtocol[Any, Any]") -> Any:
        async def _invoke(target: Any, data: Any, **kwargs: Any) -> Any:
            if hasattr(target, "run") and callable(getattr(target, "run")):
                return await target.run(data, **kwargs)
            return await target(data, **kwargs)

        class _ValidatorAgent:
            async def run(self, _data: Any, *, pipeline_context: Default.Context) -> Checklist:
                payload = {
                    "solution": pipeline_context.solution,
                    "checklist": (
                        json.loads(pipeline_context.checklist.model_dump_json())
                        if pipeline_context.checklist is not None
                        else None
                    ),
                }
                result = await _invoke(agent, json.dumps(payload))
                # Unpack the result if it has an 'output' attribute (AgentRunResult)
                unpacked_result = getattr(result, "output", result)
                pipeline_context.checklist = cast(Checklist, unpacked_result)
                return cast(Checklist, unpacked_result)

        return _ValidatorAgent()

    async def run_async(self, task: Task) -> Candidate | None:
        result: PipelineResult = await gather_result(
            self.flujo_engine,
            task.prompt,
            initial_context_data={"initial_prompt": task.prompt},
        )
        ctx = cast(Default.Context, result.final_pipeline_context)
        if ctx.solution is None or ctx.checklist is None:
            return None

        score = ratio_score(ctx.checklist)
        return Candidate(solution=ctx.solution, score=score, checklist=ctx.checklist)

    def run_sync(self, task: Task) -> Candidate | None:
        return asyncio.run(self.run_async(task))
