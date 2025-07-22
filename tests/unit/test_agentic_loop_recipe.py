"""Tests for agentic loop recipe functionality."""

import pytest
from typing import Any

from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext, PipelineResult
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from tests.conftest import create_test_flujo


class MockPlannerAgent:
    """Mock planner agent that returns commands."""

    def __init__(self, commands: list[AgentCommand]):
        self.commands = commands
        self.call_count = 0

    async def run(self, data: Any, **kwargs: Any) -> AgentCommand:
        """Return the next command in the sequence."""
        if self.call_count < len(self.commands):
            command = self.commands[self.call_count]
            self.call_count += 1
            return command
        return FinishCommand(final_answer="done")


class MockExecutorAgent:
    """Mock executor agent that simulates command execution."""

    def __init__(self, results: list[str]):
        self.results = results
        self.call_count = 0

    async def run(self, data: Any, **kwargs: Any) -> str:
        """Return the next result in the sequence."""
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
            self.call_count += 1
            return result
        return "default_result"


def test_agentic_loop_factory_creates_pipeline():
    """Test that make_agentic_loop_pipeline creates a proper pipeline."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry, max_loops=5, max_retries=2)

    assert pipeline is not None
    assert hasattr(pipeline, "steps")
    assert len(pipeline.steps) > 0


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_execution():
    """Test agentic loop pipeline execution."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry, max_loops=5, max_retries=2)
    result = await run_agentic_loop_pipeline(pipeline, "test goal")

    assert result is not None
    assert hasattr(result, "final_pipeline_context")
    assert result.final_pipeline_context is not None


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_with_multiple_commands():
    """Test agentic loop pipeline with multiple commands."""
    planner = MockPlannerAgent(
        [
            RunAgentCommand(agent_name="test", input_data="data"),
            FinishCommand(final_answer="final result"),
        ]
    )
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)
    result = await run_agentic_loop_pipeline(pipeline, "test goal")

    assert result is not None
    assert result.final_pipeline_context is not None
    assert len(result.final_pipeline_context.command_log) >= 2


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_resume():
    """Test agentic loop pipeline resume functionality."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)

    # Create a paused result
    paused_result = PipelineResult()
    paused_result.final_pipeline_context = PipelineContext(initial_prompt="test")
    paused_result.final_pipeline_context.scratchpad["status"] = "paused"

    result = await run_agentic_loop_pipeline(pipeline, "test goal", resume_from=paused_result)

    assert result is not None
    assert hasattr(result, "final_pipeline_context")


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_as_step():
    """Test agentic loop pipeline as a step in another pipeline."""
    from flujo import Step

    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)

    # Create a step from the pipeline
    step = Step.from_callable(
        lambda data, **kwargs: run_agentic_loop_pipeline(pipeline, data), name="agentic_step"
    )

    # Use it in another pipeline
    runner = create_test_flujo(step)
    from flujo.testing.utils import gather_result

    result = await gather_result(runner, "test goal")

    assert result is not None
    assert hasattr(result, "step_history")
    assert len(result.step_history) > 0


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_with_context():
    """Test agentic loop pipeline with context inheritance."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)
    result = await run_agentic_loop_pipeline(pipeline, "test goal")

    assert result is not None
    assert result.final_pipeline_context is not None
    assert result.final_pipeline_context.initial_prompt == "test goal"


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_without_context():
    """Test agentic loop pipeline without context inheritance."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)
    result = await run_agentic_loop_pipeline(pipeline, "test goal")

    assert result is not None
    assert result.final_pipeline_context is not None
