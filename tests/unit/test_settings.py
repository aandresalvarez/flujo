from flujo.infra.settings import Settings
from pydantic import SecretStr
import os
import pytest


def test_env_var_precedence(monkeypatch) -> None:
    # Legacy API key name should still be honored
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("REFLECTION_ENABLED", "false")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.reflection_enabled is False


def test_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    s = Settings()
    assert s.max_iters == 5
    assert s.k_variants == 3
    assert s.logfire_api_key is None


def test_logfire_legacy_alias(monkeypatch) -> None:
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    monkeypatch.setenv("ORCH_LOGFIRE_API_KEY", "legacy")
    s = Settings()
    assert s.logfire_api_key.get_secret_value() == "legacy"
    assert "logfire" not in s.provider_api_keys


def test_missing_api_key_allowed(monkeypatch) -> None:
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import flujo.infra.settings as settings_mod

    importlib.reload(settings_mod)
    s = Settings()
    assert isinstance(s, Settings)


def test_settings_initialization(monkeypatch) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is set in the environment; skipping test to avoid leakage.")
    # Test that constructor values are properly set when provided
    # This test verifies that the Settings class correctly handles explicit values
    test_key = SecretStr("test")
    settings = Settings(
        openai_api_key=test_key,
        google_api_key=SecretStr("test"),
        anthropic_api_key=SecretStr("test"),
        logfire_api_key=SecretStr("test"),
        reflection_enabled=True,
        reward_enabled=True,
        telemetry_export_enabled=True,
        otlp_export_enabled=True,
        default_solution_model="test",
        default_review_model="test",
        default_validator_model="test",
        default_reflection_model="test",
        agent_timeout=30,
    )
    # The test verifies that the SecretStr value is properly assigned
    assert settings.openai_api_key.get_secret_value() == "test"
    assert settings.google_api_key.get_secret_value() == "test"
    assert settings.anthropic_api_key.get_secret_value() == "test"
    assert settings.logfire_api_key.get_secret_value() == "test"


def test_test_settings() -> None:
    # This test is no longer needed since TestSettings was removed
    pass
