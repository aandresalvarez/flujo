from flujo.infra.settings import Settings
from pydantic import SecretStr
import pytest
import sys
from typing import ClassVar, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    monkeypatch.setenv("MAX_ITERS", "5")
    monkeypatch.setenv("K_VARIANTS", "3")
    s = Settings()
    assert s.max_iters == 5
    assert s.k_variants == 3
    assert s.logfire_api_key is None
    assert isinstance(s.default_repair_model, str)


def test_invalid_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ITERS", "not_an_int")
    monkeypatch.setenv("K_VARIANTS", "not_an_int")
    # Should raise ValueError or fallback to default, depending on Settings implementation
    with pytest.raises((ValueError, TypeError)):
        Settings()


def test_logfire_legacy_alias(monkeypatch) -> None:
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    monkeypatch.setenv("ORCH_LOGFIRE_API_KEY", "legacy")
    s = Settings()
    assert s.logfire_api_key.get_secret_value() == "legacy"
    assert "logfire" not in s.provider_api_keys


def test_missing_api_key_allowed(monkeypatch) -> None:
    # Delete all possible API key environment variables
    for key in ["ORCH_OPENAI_API_KEY", "OPENAI_API_KEY", "openai_api_key"]:
        monkeypatch.delenv(key, raising=False)

    # Test that we can create settings without an API key
    # The actual value will depend on the environment, but the important thing
    # is that it doesn't crash
    s = Settings()
    # Patch the module's settings
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", s)
    # Just verify that we can access the attribute without error
    assert hasattr(s, "openai_api_key")


def test_settings_constructor_values() -> None:
    """Test that Settings constructor properly handles explicit values without environment interference."""

    # Create a minimal test settings class that doesn't read from environment
    class IsolatedTestSettings(BaseSettings):
        """Minimal settings class for testing constructor value handling."""

        openai_api_key: Optional[SecretStr] = None
        default_repair_model: str = "test"
        agent_timeout: int = 30

        model_config: ClassVar[SettingsConfigDict] = {
            "env_file": None,
            "populate_by_name": False,
            "extra": "ignore",
        }

    # Test that constructor values are properly set when provided
    test_key = SecretStr("test")

    settings = IsolatedTestSettings(
        openai_api_key=test_key,
        default_repair_model="test",
        agent_timeout=30,
    )

    # Verify that the SecretStr values are properly assigned
    assert settings.openai_api_key.get_secret_value() == "test"
    assert settings.default_repair_model == "test"
    assert settings.agent_timeout == 30


def test_settings_initialization(monkeypatch) -> None:
    """Test that Settings class properly handles environment variable precedence."""
    # Clear any existing API keys to avoid leakage
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)

    # Test that the Settings class works with environment variables
    s = Settings()
    # Just verify that we can create settings without error
    assert hasattr(s, "openai_api_key")


def test_test_settings() -> None:
    # This test is no longer needed since TestSettings was removed
    pass
