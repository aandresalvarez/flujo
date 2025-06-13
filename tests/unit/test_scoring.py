import pytest
from pydantic_ai_orchestrator.domain.models import Checklist, ChecklistItem
from pydantic_ai_orchestrator.domain.scoring import (
    ratio_score,
    weighted_score,
    RewardScorer,
)
from pydantic_ai_orchestrator.infra.settings import Settings
from pydantic import SecretStr


def test_ratio_score():
    check_pass = Checklist(
        items=[ChecklistItem(description="a", passed=True), ChecklistItem(description="b", passed=True)]
    )
    check_fail = Checklist(
        items=[ChecklistItem(description="a", passed=True), ChecklistItem(description="b", passed=False)]
    )
    check_empty = Checklist(items=[])

    assert ratio_score(check_pass) == 1.0
    assert ratio_score(check_fail) == 0.5
    assert ratio_score(check_empty) == 0.0


def test_weighted_score():
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True),
            ChecklistItem(description="b", passed=False),
            ChecklistItem(description="c", passed=True),
        ]
    )
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
    from pydantic_ai_orchestrator.domain.scoring import RewardScorer, RewardModelUnavailable
    import pydantic_ai_orchestrator.domain.scoring as scoring_mod

    # --- Test Success Case ---
    enabled_settings = Settings(reward_enabled=True, openai_api_key=SecretStr("sk-test"))
    monkeypatch.setattr(scoring_mod, "settings", enabled_settings)
    RewardScorer()  # Should not raise

    # --- Test Failure Case (Missing Key) ---
    monkeypatch.delenv("orch_openai_api_key", raising=False)
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    disabled_settings = Settings(reward_enabled=True, openai_api_key=None)
    monkeypatch.setattr(scoring_mod, "settings", disabled_settings)
    with pytest.raises(RewardModelUnavailable):
        RewardScorer()


@pytest.mark.asyncio
async def test_reward_scorer_returns_float(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import pydantic_ai_orchestrator.domain.scoring as scoring_mod

    test_settings = Settings(reward_enabled=True, openai_api_key=SecretStr("sk-test"))
    monkeypatch.setattr(scoring_mod, "settings", test_settings)

    scorer = RewardScorer()
    scorer.agent.run = AsyncMock(return_value=SimpleNamespace(output=0.77))
    result = await scorer.score("x")
    assert result == 0.77


def test_reward_scorer_disabled(monkeypatch):
    from pydantic_ai_orchestrator.domain.scoring import RewardScorer, FeatureDisabled
    import pydantic_ai_orchestrator.domain.scoring as scoring_mod

    test_settings = Settings(reward_enabled=False)
    monkeypatch.setattr(scoring_mod, "settings", test_settings)

    with pytest.raises(FeatureDisabled):
        RewardScorer()


def test_weighted_score_empty_weights():
    check = Checklist(items=[ChecklistItem(description="a", passed=True)])
    assert weighted_score(check, []) == 1.0


def test_weighted_score_total_weight_zero():
    check = Checklist(items=[ChecklistItem(description="a", passed=True)])
    weights = [{"item": "a", "weight": 0.0}]
    assert weighted_score(check, weights) == 0.0


def test_redact_string_no_secret():
    from pydantic_ai_orchestrator.utils.redact import redact_string

    assert redact_string("hello world", None) == "hello world"
    assert redact_string("hello world", "") == "hello world"


def test_redact_string_secret_not_in_text():
    from pydantic_ai_orchestrator.utils.redact import redact_string

    assert redact_string("hello world", "sk-12345678") == "hello world"


def test_redact_string_secret_in_text():
    from pydantic_ai_orchestrator.utils.redact import redact_string

    assert redact_string("my key is sk-12345678abcdef", "sk-12345678abcdef") == "my key is [REDACTED]"


@pytest.mark.asyncio
async def test_reward_scorer_score_no_output(monkeypatch):
    from unittest.mock import AsyncMock
    import pydantic_ai_orchestrator.domain.scoring as scoring_mod

    test_settings = Settings(reward_enabled=True, openai_api_key=SecretStr("sk-test"))
    monkeypatch.setattr(scoring_mod, "settings", test_settings)

    scorer = RewardScorer()
    scorer.agent.run = AsyncMock(side_effect=Exception("LLM failed"))
    result = await scorer.score("x")
    assert result == 0.0


def test_ratio_score_all_passed():
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True),
            ChecklistItem(description="b", passed=True),
            ChecklistItem(description="c", passed=True),
        ]
    )
    assert ratio_score(check) == 1.0


def test_weighted_score_all_weights_present():
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True),
            ChecklistItem(description="b", passed=True),
            ChecklistItem(description="c", passed=True),
        ]
    )
    weights = [
        {"item": "a", "weight": 0.5},
        {"item": "b", "weight": 0.3},
        {"item": "c", "weight": 0.2},
    ]
    assert weighted_score(check, weights) == pytest.approx(1.0)


def test_weighted_score_invalid_weight_type():
    check = Checklist(items=[ChecklistItem(description="a", passed=True)])
    with pytest.raises(ValueError):
        weighted_score(check, ["not-a-dict"])


def test_weighted_score_missing_keys():
    check = Checklist(items=[ChecklistItem(description="a", passed=True)])
    with pytest.raises(ValueError):
        weighted_score(check, [{"item": "a"}])
