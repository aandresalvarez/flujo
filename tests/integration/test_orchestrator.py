from unittest.mock import Mock, patch, AsyncMock
import pytest
from pydantic_ai_orchestrator.application.orchestrator import Orchestrator
from pydantic_ai_orchestrator.domain.models import Task, Checklist, ChecklistItem, Candidate

@pytest.fixture
def mock_agents():
    """Fixture to create mock agents with async support."""
    review_agent = AsyncMock()
    solution_agent = AsyncMock()
    validator_agent = AsyncMock()
    reflection_agent = AsyncMock()
    return review_agent, solution_agent, validator_agent, reflection_agent

@pytest.mark.asyncio
async def test_orchestrator_short_circuits_on_perfect_score(mock_agents):
    review_agent, solution_agent, validator_agent, reflection_agent = mock_agents

    # Arrange: Setup agents to return a perfect result on the first try
    initial_checklist = Checklist(items=[ChecklistItem(description="item 1")])
    review_agent.run_async.return_value = initial_checklist
    
    solution_agent.run_async.return_value = "the perfect solution"
    
    validated_checklist = Checklist(items=[ChecklistItem(description="item 1", passed=True)])
    validator_agent.run_async.return_value = validated_checklist

    # Act
    orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent, k_variants=1)
    result_candidate = await orch.run_async(Task(prompt="do a thing"))

    # Assert
    assert result_candidate.score == 1.0
    assert result_candidate.solution == "the perfect solution"
    # The solution agent should only be called once, as the loop exits early
    solution_agent.run_async.assert_called_once()
    # Reflection agent should not be called
    reflection_agent.run_async.assert_not_called()

@pytest.mark.asyncio
async def test_orchestrator_reflection_memory_is_capped(mock_agents):
    review_agent, solution_agent, validator_agent, reflection_agent = mock_agents

    # Arrange: setup agents to consistently fail
    initial_checklist = Checklist(items=[ChecklistItem(description="item 1")])
    review_agent.run_async.return_value = initial_checklist

    solution_agent.run_async.return_value = "a failing solution"

    failed_checklist = Checklist(items=[ChecklistItem(description="item 1", passed=False, feedback="it broke")])
    validator_agent.run_async.return_value = failed_checklist

    reflection_agent.run_async.side_effect = ["reflection 1", "reflection 2", "reflection 3", "reflection 4"]

    # Act
    orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent, max_iters=4, k_variants=1)
    await orch.run_async(Task(prompt="do a thing"))
    
    # Assert: Reflection agent is called, but memory is capped
    assert reflection_agent.run_async.call_count == 3
    
    # Check that the prompt for the last solution attempt contains the first 3 reflections
    last_call_args = solution_agent.run_async.call_args_list[-1]
    prompt_arg = last_call_args.args[0]
    assert "reflection 1" in prompt_arg
    assert "reflection 2" in prompt_arg
    assert "reflection 3" in prompt_arg
    assert "reflection 4" not in prompt_arg 