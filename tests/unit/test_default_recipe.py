"""Tests for the default pipeline to ensure it handles AgentRunResult objects correctly."""

import pytest
from unittest.mock import AsyncMock

from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Task, Checklist, ChecklistItem


class MockAgentRunResult:
    """Mock AgentRunResult that simulates the structure returned by pydantic-ai agents."""

    def __init__(self, output):
        self.output = output


@pytest.mark.asyncio
async def test_default_pipeline_handles_agent_run_result():
    """Test that the default pipeline correctly unpacks AgentRunResult objects."""

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

    # Create the default pipeline
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Run the workflow
    task = Task(prompt="Write a Python function that returns 'Hello, World!'")
    result = await run_default_pipeline(pipeline, task)

    # Verify the result
    assert result is not None
    assert result.solution == "def hello(): return 'Hello, World!'"
    # The score should be 1.0 since all items in the checklist are passed=True
    assert result.score == 1.0
    assert result.checklist is not None
    assert len(result.checklist.items) == 1
    assert result.checklist.items[0].passed is True


@pytest.mark.asyncio
async def test_default_pipeline_handles_direct_results():
    """Test that the default pipeline still works with direct results (not AgentRunResult)."""

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

    # Create the default pipeline
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Run the workflow
    task = Task(prompt="Write a Python function that returns 'Hello, World!'")
    result = await run_default_pipeline(pipeline, task)

    # Verify the result
    assert result is not None
    assert result.solution == "def hello(): return 'Hello, World!'"
    # The score should be 1.0 since all items in the checklist are passed=True
    assert result.score == 1.0
    assert result.checklist is not None
    assert len(result.checklist.items) == 1
    assert result.checklist.items[0].passed is True
