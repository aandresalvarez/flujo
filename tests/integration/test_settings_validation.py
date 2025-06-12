import pytest
from pydantic import ValidationError
from pydantic_ai_orchestrator.infra.settings import Settings

def test_invalid_env_vars(monkeypatch):
    monkeypatch.delenv("ORCH_OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings() 