"""Scoring logic for pydantic-ai-orchestrator.""" 

from typing import List, Dict
from .models import Checklist, ChecklistItem
from pydantic_ai import Agent
from ..infra.settings import settings
import logfire


def ratio_score(check: Checklist) -> float:
    """
    Computes the ratio of passed items to total items in a checklist.
    """
    if not check.items:
        return 0.0
    passed = sum(1 for item in check.items if item.passed)
    return passed / len(check.items)


def weighted_score(check: Checklist, weights: List[Dict[str, float]]) -> float:
    """
    Computes a weighted score for a checklist.
    `weights` is a list of dicts, e.g., [{"item": "description", "weight": 0.7}]
    """
    if not check.items:
        return 0.0

    weight_map = {w["item"]: w["weight"] for w in weights}
    total_weight = sum(weight_map.get(item.description, 1.0) for item in check.items)
    if total_weight == 0:
        return 0.0

    score = sum(
        weight_map.get(item.description, 1.0)
        for item in check.items
        if item.passed
    )
    return score / total_weight


class RewardScorer:
    """
    Scores a solution using a reward model (LLM judge).
    Raises NotImplementedError if the required API key is not configured.
    """
    def __init__(self):
        if not settings.openai_api_key:
            raise NotImplementedError("OpenAI API key is required for RewardScorer.")
        self.agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You are a reward model. You will be given a solution and you must return a score from 0.0 to 1.0.",
            output_type=float,
        )

    @logfire.instrument("reward_score")
    def score(self, text: str) -> float:
        """Calls the LLM judge to score the given text."""
        try:
            return self.agent.run_sync(text)
        except Exception as e:
            logfire.error(f"RewardScorer failed: {e}")
            return 0.0 