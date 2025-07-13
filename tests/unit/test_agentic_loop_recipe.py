"""Tests for flujo.recipes.agentic_loop module."""

import pytest
import warnings
from typing import Any

from flujo.recipes.agentic_loop import AgenticLoop, _CommandExecutor
from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext, PipelineResult


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


def test_agentic_loop_deprecation_warning():
    warnings.simplefilter("always")
    with pytest.warns(DeprecationWarning, match="The AgenticLoop class is deprecated"):
        planner = MockPlannerAgent([FinishCommand(final_answer="done")])
        registry = {"test": MockExecutorAgent(["result"])}
        AgenticLoop(planner, registry)


def test_agentic_loop_initialization():
    """Test AgenticLoop initialization."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry, max_loops=5, max_retries=2)

    assert loop.planner_agent == planner
    assert loop.agent_registry == registry
    assert loop.max_loops == 5
    assert loop.max_retries == 2
    assert loop._pipeline is not None


def test_command_executor_initialization():
    """Test _CommandExecutor initialization."""
    registry = {"test": MockExecutorAgent(["result"])}
    executor = _CommandExecutor(registry)
    assert executor.agent_registry == registry


@pytest.mark.asyncio
async def test_command_executor_run():
    """Test _CommandExecutor.run method."""
    registry = {"test": MockExecutorAgent(["result"])}
    executor = _CommandExecutor(registry)

    # Test with a command that has an agent
    command = RunAgentCommand(agent_name="test", input_data="data")
    context = PipelineContext(initial_prompt="goal")
    result = await executor.run(command, context=context)
    # Assert on the execution_result field
    assert hasattr(result, "execution_result")
    assert result.execution_result == "result"


@pytest.mark.asyncio
async def test_command_executor_run_async():
    """Test _CommandExecutor.run_async method."""
    registry = {"test": MockExecutorAgent(["result"])}
    executor = _CommandExecutor(registry)

    command = FinishCommand(final_answer="done")
    result = await executor.run_async(command, context=PipelineContext(initial_prompt="goal"))
    assert hasattr(result, "execution_result")
    assert result.execution_result == "done"


@pytest.mark.asyncio
async def test_command_executor_run_with_context():
    """Test _CommandExecutor.run with context."""
    registry = {"test": MockExecutorAgent(["result"])}
    executor = _CommandExecutor(registry)

    command = FinishCommand(final_answer="done")
    context = PipelineContext(initial_prompt="test")

    result = await executor.run(command, context=context)
    assert hasattr(result, "execution_result")
    assert result.execution_result == "done"


@pytest.mark.asyncio
async def test_command_executor_run_with_resources():
    """Test _CommandExecutor.run with resources."""
    registry = {"test": MockExecutorAgent(["result"])}
    executor = _CommandExecutor(registry)

    command = FinishCommand(final_answer="done")
    resources = {"api_key": "test"}

    result = await executor.run(
        command, context=PipelineContext(initial_prompt="goal"), resources=resources
    )
    assert hasattr(result, "execution_result")
    assert result.execution_result == "done"


def test_agentic_loop_run_sync():
    """Test AgenticLoop.run method (sync)."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    result = loop.run("test goal")
    assert hasattr(result, "final_pipeline_context")


@pytest.mark.asyncio
async def test_agentic_loop_run_async():
    """Test AgenticLoop.run_async method."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    result = await loop.run_async("test goal")
    assert hasattr(result, "final_pipeline_context")


def test_agentic_loop_resume():
    """Test AgenticLoop.resume method (sync)."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    paused_result = PipelineResult()
    paused_result.final_pipeline_context = PipelineContext(initial_prompt="test")
    paused_result.final_pipeline_context.scratchpad["status"] = "paused"
    result = loop.resume(paused_result, "human input")
    assert hasattr(result, "final_pipeline_context")


@pytest.mark.asyncio
async def test_agentic_loop_resume_async():
    """Test AgenticLoop.resume_async method."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    paused_result = PipelineResult()
    paused_result.final_pipeline_context = PipelineContext(initial_prompt="test")
    paused_result.final_pipeline_context.scratchpad["status"] = "paused"
    result = await loop.resume_async(paused_result, "human input")
    assert hasattr(result, "final_pipeline_context")


def test_agentic_loop_as_step():
    """Test AgenticLoop.as_step method."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    step = loop.as_step("test_step")
    assert hasattr(step.agent, "run")


@pytest.mark.asyncio
async def test_agentic_loop_as_step_execution():
    """Test AgenticLoop.as_step execution."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)
    step = loop.as_step("test_step")
    result = await step.agent.run("test goal")
    # The output is in result.step_history[0].output
    assert result.step_history[0].output == "done"


def test_agentic_loop_as_step_with_context():
    """Test AgenticLoop.as_step with context inheritance."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)

    step = loop.as_step("test_step", inherit_context=True)
    assert step.name == "test_step"


def test_agentic_loop_as_step_without_context():
    """Test AgenticLoop.as_step without context inheritance."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}
    loop = AgenticLoop(planner, registry)

    step = loop.as_step("test_step", inherit_context=False)
    assert step.name == "test_step"
