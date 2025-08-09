import pytest
from flujo.domain.agent_protocol import AgentProtocol
from flujo.agents import (
    make_review_agent,
    make_solution_agent,
    make_validator_agent,
    get_reflection_agent,
    NoOpReflectionAgent,
)
from flujo.infra.settings import Settings


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch) -> None:
    """Ensure OPENAI_API_KEY is present and refresh settings for each test."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import sys

    new_settings = Settings()
    settings_module = sys.modules["flujo.infra.settings"]
    monkeypatch.setattr(settings_module, "settings", new_settings)


def test_agents_conform_to_protocol() -> None:
    assert isinstance(make_review_agent(), AgentProtocol)
    assert isinstance(make_solution_agent(), AgentProtocol)
    assert isinstance(make_validator_agent(), AgentProtocol)
    assert isinstance(get_reflection_agent(), AgentProtocol)
    assert isinstance(NoOpReflectionAgent(), AgentProtocol)
