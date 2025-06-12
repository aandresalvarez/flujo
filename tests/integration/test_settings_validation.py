import pytest
from pydantic import ValidationError
from pydantic_ai_orchestrator.infra.settings import Settings

def test_invalid_env_vars(monkeypatch):
    # from pydantic_ai_orchestrator.infra.settings import Settings  # removed redefinition
    import os
    for k in list(os.environ.keys()):
        if k in {"OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"}:
            monkeypatch.delenv(k, raising=False)
    # Patch env_file to None for this test instance
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    s = TestSettings()
    assert s.openai_api_key is None
