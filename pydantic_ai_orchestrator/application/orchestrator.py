"""Orchestration logic for pydantic-ai-orchestrator.""" 

from .temperature import temp_for_round
from ..domain.models import Task, Candidate, Checklist
from ..domain.scoring import ratio_score, weighted_score, RewardScorer
from ..infra.agents import Agent
from ..infra.settings import settings
import logfire
import random
from typing import Optional, Callable
import asyncio

class Orchestrator:
    """
    Main orchestrator for running agent-based workflows. It manages the iterative
    process of generating, validating, and reflecting on solutions to a given task.
    """
    def __init__(
        self,
        review_agent: Agent,
        solution_agent: Agent,
        validator_agent: Agent,
        reflection_agent: Agent,
        max_iters: Optional[int] = None,
        k_variants: Optional[int] = None,
        scorer: Optional[Callable] = None,
    ):
        self.review = review_agent
        self.solve = solution_agent
        self.validate = validator_agent
        self.reflect = reflection_agent

        # Fallback to settings if specific values are not provided
        self.max_iters = max_iters if max_iters is not None else settings.max_iters
        self.k_variants = k_variants if k_variants is not None else settings.k_variants
        
        # Scorer selection logic
        if scorer:
            self.scorer = scorer
        elif settings.scorer == "weighted":
            # Assuming weights are passed in task metadata for this example
            self.scorer = lambda check: weighted_score(check, check.metadata.get("weights", []))
        elif settings.scorer == "reward":
            self.scorer = RewardScorer().score
        else: # ratio is the default
            self.scorer = ratio_score

    async def run_async(self, task: Task) -> Candidate:
        """Asynchronously runs the full orchestration loop."""
        with logfire.span("orchestrator.run", task=task.prompt):
            checklist = await self.review.run_async(task.prompt)
            memory: list[str] = []
            best: Candidate | None = None

            for i in range(self.max_iters):
                with logfire.span("iteration", idx=i, memory_len=len(memory)) as iter_span:
                    prompt = f"{task.prompt}\n\nFeedback:\n{chr(10).join(memory)}"
                    temp = temp_for_round(i)
                    
                    # Generate K variants in parallel
                    solution_tasks = [self.solve.run_async(prompt, temperature=temp) for _ in range(self.k_variants)]
                    raw_solutions = await asyncio.gather(*solution_tasks)

                    iter_candidates = []
                    for raw_solution in raw_solutions:
                        with logfire.span("validation"):
                            judged_checklist = await self.validate.run_async(
                                {"solution": raw_solution, "checklist": checklist}
                            )
                            score = self.scorer(judged_checklist)
                            pass_rate = ratio_score(judged_checklist)

                            logfire.info(f"Candidate score: {score:.2f}, pass_rate: {pass_rate:.2f}")
                            iter_span.set_attribute("score", score)

                            cand = Candidate(
                                solution=raw_solution,
                                checklist=judged_checklist,
                                score=score,
                                passed=[it.description for it in judged_checklist.items if it.passed],
                                failed=[it.description for it in judged_checklist.items if not it.passed],
                            )
                            iter_candidates.append(cand)
                            if best is None or cand.score > best.score:
                                best = cand
                            
                            if pass_rate == 1.0:
                                logfire.info("Perfect solution found, exiting early.")
                                return best

                    # Reflection on the best candidate of the iteration
                    if self.reflect and best.failed and len(memory) < 3:
                        with logfire.span("reflection"):
                            reflection = await self.reflect.run_async(
                                {"failed_items": best.failed}
                            )
                            if reflection:
                                memory.append(reflection)

            return best

    def run(self, task: Task) -> Candidate:
        """Synchronously runs the full orchestration loop."""
        return asyncio.run(self.run_async(task)) 