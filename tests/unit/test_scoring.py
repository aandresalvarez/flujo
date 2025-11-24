"""Tests for scoring utilities."""

import pytest
from pydantic import SecretStr

from flujo.infra.settings import Settings
from flujo.domain.models import Checklist, ChecklistItem
from flujo.domain.scoring import weighted_score, ratio_score, RewardScorer


def monkeypatch_settings(monkeypatch, test_settings):
    """Helper function to monkeypatch settings across modules.

    This function patches both the settings instance and the settings provider
    across modules to ensure consistent test behavior. It updates:
    - flujo.infra.settings.settings: The singleton settings instance
    - flujo.infra.settings.get_settings: The settings accessor function
    - flujo.domain.interfaces._DEFAULT_SETTINGS_PROVIDER: The domain settings provider

    Args:
        monkeypatch: pytest's monkeypatch fixture for modifying module attributes
        test_settings: Settings instance to use for testing

    Usage:
        test_settings = Settings(reward_enabled=True, ...)
        monkeypatch_settings(monkeypatch, test_settings)
    """
    import sys

    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", test_settings)
    # Also monkeypatch get_settings to return our test settings
    monkeypatch.setattr(settings_module, "get_settings", lambda: test_settings)
    # Monkeypatch the domain settings provider
    import flujo.domain.interfaces as interfaces

    class _Provider:
        def __init__(self, settings):
            self._settings = settings

        def get_settings(self):
            return self._settings

    monkeypatch.setattr(interfaces, "_DEFAULT_SETTINGS_PROVIDER", _Provider(test_settings))


def test_ratio_score() -> None:
    check_pass = Checklist(
        items=[
            ChecklistItem(description="a", passed=True, feedback=None),
            ChecklistItem(description="b", passed=True, feedback=None),
        ]
    )
    check_fail = Checklist(
        items=[
            ChecklistItem(description="a", passed=True, feedback=None),
            ChecklistItem(description="b", passed=False, feedback=None),
        ]
    )
    check_empty = Checklist(items=[])

    assert ratio_score(check_pass) == 1.0
    assert ratio_score(check_fail) == 0.5
    assert ratio_score(check_empty) == 0.0


def test_weighted_score() -> None:
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True, feedback=None),
            ChecklistItem(description="b", passed=False, feedback=None),
            ChecklistItem(description="c", passed=True, feedback=None),
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


def test_reward_scorer_init_success(monkeypatch) -> None:
    from flujo.domain.scoring import RewardScorer
    from unittest.mock import Mock

    enabled_settings = Settings(
        reward_enabled=True,
        openai_api_key=SecretStr("sk-test"),
        google_api_key=None,
        anthropic_api_key=None,
        logfire_api_key=None,
        reflection_enabled=True,
        telemetry_export_enabled=False,
        otlp_export_enabled=False,
        default_solution_model="openai:gpt-4o",
        default_review_model="openai:gpt-4o",
        default_validator_model="openai:gpt-4o",
        default_reflection_model="openai:gpt-4o",
        max_iters=5,
        k_variants=3,
        reflection_limit=3,
        scorer="ratio",
        t_schedule=[1.0, 0.8, 0.5, 0.2],
        otlp_endpoint=None,
        agent_timeout=60,
    )
    monkeypatch_settings(monkeypatch, enabled_settings)
    mock_agent = Mock()
    monkeypatch.setattr("flujo.domain.scoring.Agent", mock_agent)
    RewardScorer()  # Should not raise


def test_reward_scorer_init_failure(monkeypatch) -> None:
    from flujo.domain.scoring import RewardScorer, RewardModelUnavailable
    from unittest.mock import Mock

    # Unset any possible API key env vars
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)

    disabled_settings = Settings(
        reward_enabled=True,
        openai_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
        logfire_api_key=None,
        reflection_enabled=True,
        telemetry_export_enabled=False,
        otlp_export_enabled=False,
        default_solution_model="openai:gpt-4o",
        default_review_model="openai:gpt-4o",
        default_validator_model="openai:gpt-4o",
        default_reflection_model="openai:gpt-4o",
        max_iters=5,
        k_variants=3,
        reflection_limit=3,
        scorer="ratio",
        t_schedule=[1.0, 0.8, 0.5, 0.2],
        otlp_endpoint=None,
        agent_timeout=60,
    )
    monkeypatch_settings(monkeypatch, disabled_settings)

    # Mock Agent to raise RewardModelUnavailable
    def agent_side_effect(*args, **kwargs):
        raise RewardModelUnavailable("OpenAI API key is required for RewardScorer.")

    monkeypatch.setattr("flujo.domain.scoring.Agent", Mock(side_effect=agent_side_effect))

    with pytest.raises(RewardModelUnavailable):
        RewardScorer()


@pytest.mark.asyncio
async def test_reward_scorer_returns_float(monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    monkeypatch.setenv("REWARD_ENABLED", "true")
    test_settings = Settings(
        reward_enabled=True,
        openai_api_key=SecretStr("sk-test"),
        google_api_key=None,
        anthropic_api_key=None,
        logfire_api_key=None,
        reflection_enabled=True,
        telemetry_export_enabled=False,
        otlp_export_enabled=False,
        default_solution_model="openai:gpt-4o",
        default_review_model="openai:gpt-4o",
        default_validator_model="openai:gpt-4o",
        default_reflection_model="openai:gpt-4o",
        max_iters=5,
        k_variants=3,
        reflection_limit=3,
        scorer="ratio",
        t_schedule=[1.0, 0.8, 0.5, 0.2],
        otlp_endpoint=None,
        agent_timeout=60,
    )
    monkeypatch_settings(monkeypatch, test_settings)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    scorer = RewardScorer()
    scorer.agent.run = AsyncMock(return_value=SimpleNamespace(output=0.77))
    result = await scorer.score("x")
    assert result == 0.77
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_reward_scorer_disabled(monkeypatch) -> None:
    from flujo.domain.scoring import RewardScorer, FeatureDisabled

    test_settings = Settings(
        reward_enabled=False,
        openai_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
        logfire_api_key=None,
        reflection_enabled=True,
        telemetry_export_enabled=False,
        otlp_export_enabled=False,
        default_solution_model="openai:gpt-4o",
        default_review_model="openai:gpt-4o",
        default_validator_model="openai:gpt-4o",
        default_reflection_model="openai:gpt-4o",
        max_iters=5,
        k_variants=3,
        reflection_limit=3,
        scorer="ratio",
        t_schedule=[1.0, 0.8, 0.5, 0.2],
        otlp_endpoint=None,
        agent_timeout=60,
    )
    monkeypatch_settings(monkeypatch, test_settings)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with pytest.raises(FeatureDisabled):
        RewardScorer()


def test_weighted_score_empty_weights() -> None:
    check = Checklist(items=[ChecklistItem(description="a", passed=True, feedback=None)])
    assert weighted_score(check, []) == 1.0


def test_weighted_score_total_weight_zero() -> None:
    check = Checklist(items=[ChecklistItem(description="a", passed=True, feedback=None)])
    weights = [{"item": "a", "weight": 0.0}]
    assert weighted_score(check, weights) == 0.0


def test_redact_string_no_secret() -> None:
    from flujo.utils.redact import redact_string

    assert redact_string("hello world", None) == "hello world"
    assert redact_string("hello world", "") == "hello world"


def test_redact_string_secret_not_in_text() -> None:
    from flujo.utils.redact import redact_string

    assert redact_string("hello world", "sk-12345678") == "hello world"


def test_redact_string_secret_in_text() -> None:
    from flujo.utils.redact import redact_string

    assert (
        redact_string("my key is sk-12345678abcdef", "sk-12345678abcdef") == "my key is [REDACTED]"
    )


@pytest.mark.asyncio
async def test_reward_scorer_score_no_output(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    monkeypatch.setenv("REWARD_ENABLED", "true")
    test_settings = Settings(
        reward_enabled=True,
        openai_api_key=SecretStr("sk-test"),
        google_api_key=None,
        anthropic_api_key=None,
        logfire_api_key=None,
        reflection_enabled=True,
        telemetry_export_enabled=False,
        otlp_export_enabled=False,
        default_solution_model="openai:gpt-4o",
        default_review_model="openai:gpt-4o",
        default_validator_model="openai:gpt-4o",
        default_reflection_model="openai:gpt-4o",
        max_iters=5,
        k_variants=3,
        reflection_limit=3,
        scorer="ratio",
        t_schedule=[1.0, 0.8, 0.5, 0.2],
        otlp_endpoint=None,
        agent_timeout=60,
    )
    monkeypatch_settings(monkeypatch, test_settings)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    scorer = RewardScorer()
    scorer.agent.run = AsyncMock(side_effect=Exception("LLM failed"))
    result = await scorer.score("x")
    assert result == 0.0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_ratio_score_all_passed() -> None:
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True, feedback=None),
            ChecklistItem(description="b", passed=True, feedback=None),
            ChecklistItem(description="c", passed=True, feedback=None),
        ]
    )
    assert ratio_score(check) == 1.0


def test_weighted_score_all_weights_present() -> None:
    check = Checklist(
        items=[
            ChecklistItem(description="a", passed=True, feedback=None),
            ChecklistItem(description="b", passed=True, feedback=None),
            ChecklistItem(description="c", passed=True, feedback=None),
        ]
    )
    weights = [
        {"item": "a", "weight": 0.5},
        {"item": "b", "weight": 0.3},
        {"item": "c", "weight": 0.2},
    ]
    assert weighted_score(check, weights) == pytest.approx(1.0)


def test_weighted_score_invalid_weight_type() -> None:
    check = Checklist(items=[ChecklistItem(description="a", passed=True, feedback=None)])
    with pytest.raises(ValueError):
        weighted_score(check, ["not-a-dict"])


def test_weighted_score_missing_keys() -> None:
    check = Checklist(items=[ChecklistItem(description="a", passed=True, feedback=None)])
    with pytest.raises(ValueError):
        weighted_score(check, [{"item": "a"}])
