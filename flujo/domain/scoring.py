"""Scoring utilities for flujo."""

from typing import List, Any
from .models import Checklist
from pydantic_ai import Agent
import os
from .interfaces import get_settings_provider, get_telemetry_sink, TelemetrySink
from ..exceptions import FeatureDisabled, RewardModelUnavailable
from flujo.type_definitions.common import JSONObject


def ratio_score(check: Checklist) -> float:
    """
    Computes the ratio of passed items to total items in a checklist.
    """
    if not check.items:
        return 0.0
    passed = sum(1 for item in check.items if item.passed)
    return passed / len(check.items)


def weighted_score(check: Checklist, weights: List[JSONObject]) -> float:
    """
    Computes a weighted score for a checklist.
    `weights` is a list of dicts, e.g., [{"item": "description", "weight": 0.7}]
    """
    if not check.items:
        return 0.0

    if not isinstance(weights, list):
        raise ValueError("weights must be a list of dicts with 'item' and 'weight'")

    weight_map: dict[str, float] = {}
    for w in weights:
        if (
            not isinstance(w, dict)
            or not isinstance(w.get("item"), str)
            or not isinstance(w.get("weight"), (int, float))
        ):
            raise ValueError("weights must be a list of dicts with 'item' and 'weight'")
        weight_map[str(w["item"])] = float(w["weight"])
    total_weight = sum(weight_map.get(item.description, 1.0) for item in check.items)
    if total_weight == 0:
        return 0.0

    score = sum(weight_map.get(item.description, 1.0) for item in check.items if item.passed)
    return score / total_weight


class RewardScorer:
    """
    Scores a solution using a reward model (LLM judge).
    Raises errors if the feature is disabled or the API key is missing.
    """

    def __init__(self) -> None:
        self._telemetry: TelemetrySink = get_telemetry_sink()

        # Always fetch the current settings via the domain interface to support injection in tests
        settings_provider = get_settings_provider()
        settings = settings_provider.get_settings()
        if settings is None:
            raise RewardModelUnavailable("Settings provider did not return settings.")
        if not getattr(settings, "reward_enabled", False):
            raise FeatureDisabled("RewardScorer is disabled by settings.")
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise RewardModelUnavailable("OpenAI API key is required for RewardScorer.")

        # The Agent constructor's type hints are not strict enough for mypy strict mode.
        # See: https://github.com/pydantic/pydantic-ai/issues (file an issue if not present)
        os.environ.setdefault("OPENAI_API_KEY", api_key.get_secret_value())
        self.agent = Agent(
            "openai:gpt-4o-mini",
            system_prompt="You are a reward model. You return a single float score from 0.0 to 1.0.",
            output_type=float,
        )

    async def score(self, text: str) -> float:
        """Calls the LLM judge to score the given text, returning its raw output. Async."""
        try:
            # The output of a pydantic-ai agent run is the parsed model, not an AgentResult.
            with self._telemetry.span("reward_score"):
                result: Any = await self.agent.run(text)
            return float(result.output)
        except Exception as e:
            try:
                self._telemetry.error(f"RewardScorer failed: {e}")
            except Exception:
                pass
            return 0.0
