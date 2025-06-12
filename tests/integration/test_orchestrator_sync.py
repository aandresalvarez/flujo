from unittest.mock import MagicMock
import pytest
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.domain.models import Task, Checklist, ChecklistItem, Candidate
from types import SimpleNamespace

@pytest.fixture
def mock_agents():
    """Fixture to create mock agents for sync testing."""
    # We don't need async mocks here, but the interface must be awaitable
    async def async_return(value):
        return value
    
    # Simulate AgentResult with .output property
    def agent_result(val):
        return SimpleNamespace(output=val)
    
    review_agent = MagicMock()
    review_agent.run.side_effect = lambda _: async_return(agent_result(Checklist(items=[ChecklistItem(description="item 1", passed=True)])))
    
    solution_agent = MagicMock()
    solution_agent.run.side_effect = lambda _prompt, temperature: async_return(agent_result("the sync solution"))
    
    validator_agent = MagicMock()
    validator_agent.run.side_effect = lambda _: async_return(agent_result(Checklist(items=[ChecklistItem(description="item 1", passed=True)])))

    reflection_agent = MagicMock()
    return review_agent, solution_agent, validator_agent, reflection_agent

def test_orchestrator_sync_happy_path(mock_agents, monkeypatch):
    """
    Tests that the synchronous `run_sync` method successfully orchestrates
    a simple, one-iteration flow and returns the expected candidate.
    """
    review_agent, solution_agent, validator_agent, reflection_agent = mock_agents
    
    # Disable reflection for this simple test
    monkeypatch.setattr("pydantic_ai_orchestrator.infra.settings.settings.reflection_enabled", False)

    # Act
    orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent, k_variants=1, max_iters=1)
    result_candidate = orch.run_sync(Task(prompt="do a thing"))

    # Assert
    assert isinstance(result_candidate, Candidate)
    assert result_candidate.solution == "the sync solution"
    assert result_candidate.score == 1.0
    
    review_agent.run.assert_called_once()
    solution_agent.run.assert_called_once()
    validator_agent.run.assert_called_once()
    reflection_agent.run.assert_not_called()

def test_orchestrator_sync_returns_score_1(mock_agents):
    orch = Orchestrator(*mock_agents, k_variants=1, max_iters=1)
    res = orch.run_sync(Task(prompt="dummy"))
    assert res.score == 1.0 