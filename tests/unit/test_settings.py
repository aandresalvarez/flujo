import pytest
from pydantic import ValidationError
from pydantic_ai_orchestrator.infra.settings import Settings

def test_env_var_precedence(monkeypatch):
    monkeypatch.setenv("orch_openai_api_key", "sk-test")
    monkeypatch.setenv("orch_reflection_enabled", "false")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.reflection_enabled is False

def test_defaults(monkeypatch):
    monkeypatch.delenv("LOGFIRE_API_KEY", raising=False)
    s = Settings()
    assert s.max_iters == 5
    assert s.k_variants == 3
    assert s.logfire_api_key is None

def test_missing_api_key_allowed(monkeypatch):
    monkeypatch.delenv("orch_openai_api_key", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    import pydantic_ai_orchestrator.infra.settings as settings_mod
    importlib.reload(settings_mod)
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    s = TestSettings()
    assert isinstance(s, Settings)

