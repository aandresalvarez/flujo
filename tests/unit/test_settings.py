from flujo.infra.settings import Settings
from pydantic import SecretStr
import os


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


def test_missing_api_key_allowed(monkeypatch) -> None:
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import flujo.infra.settings as settings_mod

    importlib.reload(settings_mod)
    s = Settings()
    assert isinstance(s, Settings)


def test_settings_initialization(monkeypatch) -> None:
    # Unset env var so constructor value is used
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    settings = Settings(
        openai_api_key=SecretStr("test"),
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
        agent_timeout=30
    )
    assert settings.openai_api_key.get_secret_value() == "test"


def test_test_settings() -> None:
    # This test is no longer needed since TestSettings was removed
    pass
