"""
Integration test for the refactored, context-driven Default recipe.

This test verifies that the `Default` recipe correctly uses the `Flujo` engine
and a `Typed Pipeline Context` to manage the data flow between agents,
ensuring each agent receives the appropriate inputs.
"""

import pytest
from unittest.mock import AsyncMock, patch

from flujo.recipes.default import Default
from flujo.domain.models import Task, Candidate, Checklist, ChecklistItem


@pytest.fixture
def mock_agents() -> dict[str, AsyncMock]:
    """Provides a dictionary of mocked agents for the Default recipe."""
    # The review agent returns a simple checklist.
    review_agent = AsyncMock()
    review_agent.run = AsyncMock(
        return_value=Checklist(items=[ChecklistItem(description="item 1")])
    )

    # The solution agent returns a simple string solution.
    solution_agent = AsyncMock()
    solution_agent.run = AsyncMock(return_value="The final solution.")

    # The validator agent returns the checklist, simulating it has been filled out.
    validator_agent = AsyncMock()
    validator_agent.run = AsyncMock(
        return_value=Checklist(items=[ChecklistItem(description="item 1", passed=True)])
    )
    reflection_agent = AsyncMock()
    reflection_agent.run = AsyncMock(return_value="Reflection complete.")
    return {
        "review": review_agent,
        "solution": solution_agent,
        "validator": validator_agent,
        "reflection": reflection_agent,
    }


@pytest.mark.asyncio
async def test_default_recipe_data_flow(mock_agents: dict[str, AsyncMock]):
    """Tests that the Default recipe orchestrates the agents with the correct data flow."""
    # Instantiate the Default recipe with our mocked agents.
    orch = Default(
        review_agent=mock_agents["review"],
        solution_agent=mock_agents["solution"],
        validator_agent=mock_agents["validator"],
        reflection_agent=mock_agents["reflection"],
    )

    task = Task(prompt="Test prompt")

    # Patch the `run_async` method of the internal Flujo engine to inspect its inputs.
    with patch.object(
        orch.flujo_engine, "run_async", wraps=orch.flujo_engine.run_async
    ) as mock_flujo_run:
        result = await orch.run_async(task)
        mock_flujo_run.assert_called_once()
        call_args = mock_flujo_run.call_args
        assert call_args.kwargs["initial_context_data"]["initial_prompt"] == "Test prompt"

    # Verify that each mocked agent was called with the correct arguments.
    mock_agents["review"].run.assert_called_once_with("Test prompt")
    mock_agents["solution"].run.assert_called_once_with("Test prompt")

    mock_agents["validator"].run.assert_called_once()
    validator_input_dict = mock_agents["validator"].run.call_args.args[0]
    assert validator_input_dict["solution"] == "The final solution."
    assert validator_input_dict["checklist"].items[0].description == "item 1"

    assert isinstance(result, Candidate)
    assert result.solution == "The final solution."
    assert result.score == 1.0
    assert result.checklist is not None
    assert result.checklist.items[0].passed is True
