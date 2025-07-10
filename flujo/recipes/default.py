"""Default recipe for Review → Solution → Validate workflows.

DEPRECATED: This class-based approach is deprecated. Use the new factory functions
for better transparency, composability, and future YAML/AI support:

- Use `make_default_pipeline()` to create a Pipeline object
- Use `run_default_pipeline()` to execute the pipeline

See `flujo.recipes.factories` for the new approach.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, TYPE_CHECKING, cast
import warnings

from flujo.domain.models import PipelineContext

if TYPE_CHECKING:  # pragma: no cover - used for typing only
    from ..infra.agents import AsyncAgentProtocol

from ..domain.dsl.step import Step
from ..domain.models import Candidate, PipelineResult, Task, Checklist
from ..domain.scoring import ratio_score
from ..application.runner import Flujo
from ..testing.utils import gather_result


class Default:
    """Default recipe for Review → Solution → Validate workflows.

    DEPRECATED: This class-based approach is deprecated. Use the new factory functions
    for better transparency, composability, and future YAML/AI support:

    - Use `make_default_pipeline()` to create a Pipeline object
    - Use `run_default_pipeline()` to execute the pipeline

    See `flujo.recipes.factories` for the new approach.
    """

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
        warnings.warn(
            "The Default class is deprecated. Use make_default_pipeline() and run_default_pipeline() "
            "from flujo.recipes.factories for better transparency, composability, and future YAML/AI support.",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = max_iters, k_variants, reflection_limit

        async def _invoke(target: Any, data: Any, **kwargs: Any) -> Any:
            if hasattr(target, "run") and callable(getattr(target, "run")):
                return await target.run(data, **kwargs)
            return await target(data, **kwargs)

        class ReviewWrapper:
            async def run(self, data: Any, *, context: PipelineContext) -> Any:
                result = await _invoke(review_agent, data, context=context)
                checklist = cast(Checklist, getattr(result, "output", result))
                context.scratchpad["checklist"] = checklist
                return cast(str, data)

            async def run_async(self, data: Any, *, context: PipelineContext) -> Any:
                return await self.run(data, context=context)

        class SolutionWrapper:
            async def run(self, data: Any, *, context: PipelineContext) -> Any:
                result = await _invoke(solution_agent, data, context=context)
                solution = cast(str, getattr(result, "output", result))
                context.scratchpad["solution"] = solution
                return solution

            async def run_async(self, data: Any, *, context: PipelineContext) -> Any:
                return await self.run(data, context=context)

        class ValidatorWrapper:
            async def run(self, _data: Any, *, context: PipelineContext) -> Any:
                payload = {
                    "solution": context.scratchpad.get("solution"),
                    "checklist": context.scratchpad.get("checklist"),
                }
                result = await _invoke(validator_agent, payload, context=context)
                validated = cast(Checklist, getattr(result, "output", result))
                context.scratchpad["checklist"] = validated
                return validated

            async def run_async(self, _data: Any, *, context: PipelineContext) -> Any:
                return await self.run(_data, context=context)

        pipeline = (
            Step.review(cast("AsyncAgentProtocol[Any, Any]", ReviewWrapper()), max_retries=3)
            >> Step.solution(cast("AsyncAgentProtocol[Any, Any]", SolutionWrapper()), max_retries=3)
            >> Step.validate_step(
                cast("AsyncAgentProtocol[Any, Any]", ValidatorWrapper()), max_retries=3
            )
        )

        if reflection_agent is not None:

            async def reflection_step(_: Any, *, context: PipelineContext) -> str:
                payload = {
                    "solution": context.scratchpad.get("solution"),
                    "checklist": context.scratchpad.get("checklist"),
                }
                result = await _invoke(reflection_agent, payload)
                reflection = cast(str, getattr(result, "output", result))
                context.scratchpad["reflection"] = reflection
                return reflection

            pipeline = pipeline >> Step.from_callable(
                reflection_step, name="reflection", max_retries=3
            )

        self.flujo_engine = Flujo(pipeline, context_model=PipelineContext)

    async def run_async(self, task: Task) -> Candidate | None:
        result: PipelineResult[PipelineContext] = await gather_result(
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
