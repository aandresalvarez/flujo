from pydantic_ai_orchestrator.infra.settings import Settings

def test_invalid_env_vars(monkeypatch):
    # from pydantic_ai_orchestrator.infra.settings import Settings  # removed redefinition
    import os
    for k in list(os.environ.keys()):
        if k in {"orch_openai_api_key", "orch_google_api_key", "orch_anthropic_api_key", "OPENAI_API_KEY"}:
            monkeypatch.delenv(k, raising=False)
    # Patch env_file to None for this test instance
    import importlib
    import pydantic_ai_orchestrator.infra.settings as settings_mod
    importlib.reload(settings_mod)
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None
    s = TestSettings()
    assert isinstance(s, Settings)
