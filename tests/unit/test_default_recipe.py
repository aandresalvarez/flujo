"""Tests for the Default recipe to ensure it handles AgentRunResult objects correctly."""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes import Default
from flujo.domain.models import Task, Checklist, ChecklistItem


class MockAgentRunResult:
    """Mock AgentRunResult that simulates the structure returned by pydantic-ai agents."""

    def __init__(self, output):
        self.output = output


@pytest.mark.asyncio
async def test_default_recipe_handles_agent_run_result():
    """Test that the Default recipe correctly unpacks AgentRunResult objects."""

    # Create mock agents that return AgentRunResult objects
    review_agent = AsyncMock()
    review_agent.run.return_value = MockAgentRunResult(
        Checklist(items=[ChecklistItem(description="Test criterion", passed=None)])
    )

    solution_agent = AsyncMock()
    solution_agent.run.return_value = MockAgentRunResult("def hello(): return 'Hello, World!'")

    validator_agent = AsyncMock()
    validator_agent.run.return_value = MockAgentRunResult(
        Checklist(items=[ChecklistItem(description="Test criterion", passed=True)])
    )

    # Create the Default recipe
    orch = Default(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Run the workflow
    task = Task(prompt="Write a Python function that returns 'Hello, World!'")
    result = await orch.run_async(task)

    # Verify the result
    assert result is not None
    assert result.solution == "def hello(): return 'Hello, World!'"
    assert result.score == 1.0
    assert result.checklist is not None
    assert len(result.checklist.items) == 1
    assert result.checklist.items[0].passed is True


@pytest.mark.asyncio
async def test_default_recipe_handles_direct_results():
    """Test that the Default recipe still works with direct results (not AgentRunResult)."""

    # Create mock agents that return direct results
    review_agent = AsyncMock()
    review_agent.run.return_value = Checklist(
        items=[ChecklistItem(description="Test criterion", passed=None)]
    )

    solution_agent = AsyncMock()
    solution_agent.run.return_value = "def hello(): return 'Hello, World!'"

    validator_agent = AsyncMock()
    validator_agent.run.return_value = Checklist(
        items=[ChecklistItem(description="Test criterion", passed=True)]
    )

    # Create the Default recipe
    orch = Default(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Run the workflow
    task = Task(prompt="Write a Python function that returns 'Hello, World!'")
    result = await orch.run_async(task)

    # Verify the result
    assert result is not None
    assert result.solution == "def hello(): return 'Hello, World!'"
    assert result.score == 1.0
    assert result.checklist is not None
    assert len(result.checklist.items) == 1
    assert result.checklist.items[0].passed is True
