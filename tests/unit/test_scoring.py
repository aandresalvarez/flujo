import pytest
from pydantic_ai_orchestrator.domain.models import Checklist, ChecklistItem
from pydantic_ai_orchestrator.domain.scoring import ratio_score, weighted_score, RewardScorer
from pydantic_ai_orchestrator.infra.settings import Settings
from pydantic_ai_orchestrator.domain.scoring import RewardModelUnavailable, FeatureDisabled
from pydantic import ValidationError

def test_ratio_score():
    check_pass = Checklist(items=[ChecklistItem(description="a", passed=True), ChecklistItem(description="b", passed=True)])
    check_fail = Checklist(items=[ChecklistItem(description="a", passed=True), ChecklistItem(description="b", passed=False)])
    check_empty = Checklist(items=[])
    
    assert ratio_score(check_pass) == 1.0
    assert ratio_score(check_fail) == 0.5
    assert ratio_score(check_empty) == 0.0

def test_weighted_score():
    check = Checklist(items=[
        ChecklistItem(description="a", passed=True),
        ChecklistItem(description="b", passed=False),
        ChecklistItem(description="c", passed=True),
    ])
    weights = [
        {"item": "a", "weight": 0.5},
        {"item": "b", "weight": 0.3},
        {"item": "c", "weight": 0.2},
    ]
    # (0.5 * 1 + 0.3 * 0 + 0.2 * 1) / (0.5 + 0.3 + 0.2) = 0.7 / 1.0
    assert weighted_score(check, weights) == pytest.approx(0.7)

    # Test with missing weight, defaults to 1.0
    weights_missing = [{"item": "a", "weight": 0.5}]
    # (0.5 * 1 + 1.0 * 0 + 1.0 * 1) / (0.5 + 1.0 + 1.0) = 1.5 / 2.5 = 0.6
    assert weighted_score(check, weights_missing) == pytest.approx(0.6)

def test_reward_scorer_init(monkeypatch):
    from pydantic_ai_orchestrator.infra.settings import Settings
    from pydantic_ai_orchestrator.domain.scoring import RewardScorer
    from pydantic import ValidationError
    # Should work with key
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-test")
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    import pydantic_ai_orchestrator.domain.scoring as scoring_mod
    scoring_mod.settings = TestSettings()
    RewardScorer()
    # Should fail without key
    monkeypatch.delenv("ORCH_OPENAI_API_KEY")
    with pytest.raises(ValidationError):
        scoring_mod.settings = TestSettings()
        RewardScorer()

@pytest.mark.asyncio
async def test_reward_scorer_returns_float(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-test")
    scorer = RewardScorer()
    async def async_run(*args, **kwargs):
        return SimpleNamespace(output=0.77)
    scorer.agent.run = AsyncMock(side_effect=async_run)
    result = await scorer.score("x")
    assert result == 0.77 