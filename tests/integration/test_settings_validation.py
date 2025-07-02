from flujo.infra.settings import Settings


def test_invalid_env_vars(monkeypatch):
    # from flujo.infra.settings import Settings  # removed redefinition
    import os

    for k in list(os.environ.keys()):
        if k in {
            "ORCH_OPENAI_API_KEY",
            "ORCH_GOOGLE_API_KEY",
            "ORCH_ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "orch_openai_api_key",
            "orch_google_api_key",
            "orch_anthropic_api_key",
        }:
            monkeypatch.delenv(k, raising=False)

    # Patch env_file to None for this test instance
    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = None

    s = TestSettings()
    assert isinstance(s, Settings)
