import pytest
from pydantic import ValidationError
from pydantic_ai_orchestrator.infra.settings import Settings

def test_env_var_precedence(monkeypatch):
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ORCH_REFLECTION_ENABLED", "false")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.reflection_enabled is False

def test_defaults(monkeypatch):
    monkeypatch.delenv("ORCH_LOGFIRE_API_KEY", raising=False)
    monkeypatch.setenv("ORCH_OPENAI_API_KEY", "sk-test")
    s = Settings()
    assert s.max_iters == 5
    assert s.k_variants == 3
    assert s.logfire_api_key is None

def test_validation_error(monkeypatch):
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    with pytest.raises(ValidationError):
        TestSettings() 