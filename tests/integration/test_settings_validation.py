import pytest
from pydantic import ValidationError
from pydantic_ai_orchestrator.infra.settings import Settings, SettingsError

def test_invalid_env_vars(monkeypatch):
    from pydantic_ai_orchestrator.infra.settings import Settings
    import os
    for k in list(os.environ.keys()):
        if k.startswith("ORCH_"):
            monkeypatch.delenv(k, raising=False)
    # Patch env_file to None for this test instance
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    with pytest.raises(ValidationError):
        TestSettings() 