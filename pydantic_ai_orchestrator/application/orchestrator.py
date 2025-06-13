"""Orchestration logic for pydantic-ai-orchestrator.""" 

from .temperature import temp_for_round
from ..domain.models import Task, Candidate, Checklist
from ..domain.scoring import ratio_score, weighted_score, RewardScorer
from ..exceptions import OrchestratorRetryError, FeatureDisabled
from ..infra.settings import settings
from ..domain.agent_protocol import AgentProtocol
import logfire
from typing import Optional, Any
import asyncio
from ..utils.redact import redact_string

class Orchestrator:
    """
    Main orchestrator for running agent-based workflows. It manages the iterative
    process of generating, validating, and reflecting on solutions to a given task.
    """
    def __init__(
        self,
        review_agent: AgentProtocol[str, Any],
        solution_agent: AgentProtocol[str, Any],
        validator_agent: AgentProtocol[dict[str, Any], Any],
        reflection_agent: AgentProtocol[dict[str, Any], Any],
        max_iters: Optional[int] = None,
        k_variants: Optional[int] = None,
        reflection_limit: Optional[int] = None,
    ):
        self.review = review_agent
        self.solve = solution_agent
        self.validate = validator_agent
        self.reflect = reflection_agent

        self.max_iters = max_iters if max_iters is not None else settings.max_iters
        self.k_variants = k_variants if k_variants is not None else settings.k_variants
        self.reflection_limit = reflection_limit if reflection_limit is not None else settings.reflection_limit

        self.reward_scorer = None
        if settings.scorer == "reward":
            try:
                self.reward_scorer = RewardScorer()  # type: ignore[misc]
            except FeatureDisabled:
                pass # It's ok if it's disabled, we just won't use it.

    async def _run_internal(self, task: Task) -> Candidate | None:
        with logfire.span("orchestrator.run", task=task.prompt):
            try:
                checklist_result = await asyncio.wait_for(self.review.run(task.prompt), timeout=settings.agent_timeout)
                checklist = getattr(checklist_result, 'output', checklist_result)
                if not isinstance(checklist, Checklist):
                    raise OrchestratorRetryError(f"Review agent did not return a Checklist instance. Got: {type(checklist)} - {checklist}")
            except Exception as e:
                msg = f"Review agent failed after all retries: {e}"
                msg = redact_string(msg, settings.openai_api_key.get_secret_value() if settings.openai_api_key else "")
                msg = redact_string(msg, settings.logfire_api_key.get_secret_value() if settings.logfire_api_key else "")
                raise OrchestratorRetryError(msg)

            memory: list[str] = []
            best: Candidate | None = None
            for i in range(self.max_iters):
                with logfire.span("iteration", idx=i, memory_len=len(memory)) as iter_span:
                    prompt = f"{task.prompt}\n\nFeedback:\n{chr(10).join(memory)}"
                    tasks = [asyncio.create_task(self.solve.run(prompt, temperature=temp_for_round(i))) for _ in range(self.k_variants)]
                    try:
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED, timeout=settings.agent_timeout)
                        if pending:
                            for t in pending:
                                t.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                            raise asyncio.TimeoutError("One or more solution variants timed out.")
                        variant_results = [t.result() for t in done]
                        variants = [getattr(v, 'output', v) for v in variant_results]
                    except Exception as e:
                        msg = f"Solution generation failed for iteration {i}: {e}"
                        msg = redact_string(msg, settings.openai_api_key.get_secret_value() if settings.openai_api_key else "")
                        msg = redact_string(msg, settings.logfire_api_key.get_secret_value() if settings.logfire_api_key else "")
                        logfire.warn(msg)
                        continue

                    for raw_solution in variants:
                        # Guard: checklist must be a Checklist instance
                        if not isinstance(checklist, Checklist):
                            logfire.warn("Checklist is not a Checklist instance; skipping validation for this candidate.")
                            continue
                        try:
                            judged_result = await asyncio.wait_for(self.validate.run({"solution": raw_solution, "checklist": checklist.model_copy(deep=True)}), timeout=settings.agent_timeout)
                            judged_checklist = getattr(judged_result, 'output', judged_result)
                            # Second guard: ensure judged_checklist is a Checklist instance
                            if not isinstance(judged_checklist, Checklist):
                                logfire.warn("Judged checklist is not a Checklist instance; skipping scoring for this candidate.")
                                continue
                        except Exception as e:
                            msg = f"Validation failed for a candidate: {e}"
                            msg = redact_string(msg, settings.openai_api_key.get_secret_value() if settings.openai_api_key else "")
                            msg = redact_string(msg, settings.logfire_api_key.get_secret_value() if settings.logfire_api_key else "")
                            logfire.warn(msg)
                            continue
                        score = await self._calculate_score(judged_checklist, raw_solution, task)
                        iter_span.set_attribute("score", score)
                        cand = Candidate(
                            solution=raw_solution,
                            checklist=judged_checklist,
                            score=score,
                        )
                        if best is None or cand.score > best.score:
                            best = cand
                        if score == 1.0:
                            return best
                    failed_items = (
                        [it.description for it in best.checklist.items if not it.passed]
                        if best and best.checklist
                        else []
                    )
                    if self.reflect and best and failed_items and len(memory) < self.reflection_limit:
                        reflection_result = await self.reflect.run({"failed_items": failed_items})
                        reflection = getattr(reflection_result, 'output', reflection_result)
                        if reflection:
                            memory.append(reflection)
            return best

    async def run_async(self, task: Task) -> Candidate | None:
        return await self._run_internal(task)

    def run_sync(self, task: Task) -> Candidate | None:
        return asyncio.run(self._run_internal(task))

    async def _calculate_score(self, checklist: Checklist, solution: str, task: Task) -> float:
        if settings.scorer == "weighted":
            weights = task.metadata.get("weights", [])
            return weighted_score(checklist, weights)
        if settings.scorer == "reward" and self.reward_scorer:
            return await self.reward_scorer.score(solution)
        return ratio_score(checklist) 