"""Scoring logic for pydantic-ai-orchestrator."""

from typing import List, Dict
from .models import Checklist
from pydantic_ai import Agent
import pydantic_ai_orchestrator.infra.settings as settings_mod
from pydantic_ai_orchestrator.infra.telemetry import logfire
from ..exceptions import RewardModelUnavailable, FeatureDisabled


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

    if not isinstance(weights, list):
        raise ValueError("weights must be a list of dicts with 'item' and 'weight'")

    weight_map = {}
    for w in weights:
        if not isinstance(w, dict) or "item" not in w or "weight" not in w:
            raise ValueError("weights must be a list of dicts with 'item' and 'weight'")
        weight_map[w["item"]] = w["weight"]
    total_weight = sum(weight_map.get(item.description, 1.0) for item in check.items)
    if total_weight == 0:
        return 0.0

    score = sum(
        weight_map.get(item.description, 1.0) for item in check.items if item.passed
    )
    return score / total_weight


class RewardScorer:
    """
    Scores a solution using a reward model (LLM judge).
    Raises errors if the feature is disabled or the API key is missing.
    """

    def __init__(self):
        # Always fetch the current settings from the module to support monkeypatching in tests
        settings = getattr(settings_mod, "settings")
        if not settings.reward_enabled:
            raise FeatureDisabled("RewardScorer is disabled by settings.")
        if not settings.openai_api_key:
            raise RewardModelUnavailable("OpenAI API key is required for RewardScorer.")

        self.agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You are a reward model. You return a single float score from 0.0 to 1.0.",
            output_type=float,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    @logfire.instrument("reward_score")
    async def score(self, text: str) -> float:
        """Calls the LLM judge to score the given text, returning its raw output. Async."""
        try:
            # The output of a pydantic-ai agent run is the parsed model, not an AgentResult
            result = await self.agent.run(text)
            return result.output
        except Exception as e:
            logfire.error(f"RewardScorer failed: {e}")
            return 0.0
