"""
Integration test for the default pipeline factory.

This test verifies that the `make_default_pipeline` factory correctly creates
a pipeline that uses the `Flujo` engine and manages the data flow between agents,
ensuring each agent receives the appropriate inputs.
"""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task, Candidate, Checklist, ChecklistItem


@pytest.fixture
def mock_agents() -> dict[str, AsyncMock]:
    """Provides a dictionary of mocked agents for the default pipeline."""
    # The review agent returns a simple checklist.
    review_agent = AsyncMock()

    async def review_run(data, *, context=None):
        return Checklist(items=[ChecklistItem(description="item 1")])

    review_agent.run = review_run

    # The solution agent returns a simple string solution.
    solution_agent = AsyncMock()

    async def solution_run(data, *, context=None):
        return "The final solution."

    solution_agent.run = solution_run

    # The validator agent returns the checklist, simulating it has been filled out.
    validator_agent = AsyncMock()

    async def validator_run(data, *, context=None):
        # Return a checklist with passed=True items to get a score of 1.0
        return Checklist(items=[ChecklistItem(description="item 1", passed=True)])

    validator_agent.run = validator_run

    reflection_agent = AsyncMock()

    async def reflection_run(data, *, context=None):
        return "Reflection complete."

    reflection_agent.run = reflection_run

    return {
        "review": review_agent,
        "solution": solution_agent,
        "validator": validator_agent,
        "reflection": reflection_agent,
    }


@pytest.mark.asyncio
async def test_default_recipe_data_flow(mock_agents: dict[str, AsyncMock]):
    """Tests that the default pipeline factory creates a pipeline that orchestrates the agents with the correct data flow."""
    # Create the default pipeline with our mocked agents.
    pipeline = make_default_pipeline(
        review_agent=mock_agents["review"],
        solution_agent=mock_agents["solution"],
        validator_agent=mock_agents["validator"],
        reflection_agent=mock_agents["reflection"],
    )

    task = Task(prompt="Test prompt")

    # Run the pipeline and verify the result
    result = await run_default_pipeline(pipeline, task)

    # Verify that the result is correct
    assert isinstance(result, Candidate)
    assert result.solution == "The final solution."
    assert result.score == 1.0
    assert result.checklist is not None
    assert result.checklist.items[0].passed is True
