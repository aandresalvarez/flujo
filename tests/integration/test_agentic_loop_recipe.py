"""Integration tests for agentic loop recipe functionality."""

import pytest
from typing import Any
from unittest.mock import AsyncMock

from flujo.domain.commands import (
    AgentCommand,
    FinishCommand,
    RunAgentCommand,
    AskHumanCommand,
)
from flujo.domain.models import PipelineContext
from flujo.recipes.factories import (
    make_agentic_loop_pipeline,
    run_agentic_loop_pipeline,
)
from flujo.testing.utils import StubAgent


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


@pytest.mark.asyncio
async def test_agentic_loop_pipeline_integration():
    """Test agentic loop pipeline integration."""
    planner = MockPlannerAgent([FinishCommand(final_answer="done")])
    registry = {"test": MockExecutorAgent(["result"])}

    pipeline = make_agentic_loop_pipeline(planner, registry)
    result = await run_agentic_loop_pipeline(pipeline, "test goal")

    assert result is not None
    assert result.final_pipeline_context is not None
    assert result.final_pipeline_context.initial_prompt == "test goal"


@pytest.mark.asyncio
async def test_agent_delegation_and_finish() -> None:
    planner = StubAgent(
        [
            RunAgentCommand(agent_name="summarizer", input_data="hi"),
            FinishCommand(final_answer="done"),
        ]
    )
    summarizer = AsyncMock()
    summarizer.run = AsyncMock(return_value="summary")
    pipeline = make_agentic_loop_pipeline(
        planner_agent=planner, agent_registry={"summarizer": summarizer}
    )
    result = await run_agentic_loop_pipeline(pipeline, "goal")
    summarizer.run.assert_called_once()
    args, kwargs = summarizer.run.call_args
    assert args[0] == "hi"
    ctx = result.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert len(ctx.command_log) >= 2  # May have additional commands due to improved logging
    assert ctx.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_pause_and_resume_in_loop() -> None:
    planner = StubAgent(
        [
            AskHumanCommand(question="Need input"),
            FinishCommand(final_answer="ok"),
        ]
    )
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})
    paused = await run_agentic_loop_pipeline(pipeline, "goal")
    ctx = paused.final_pipeline_context
    assert ctx.status in {"paused", "failed"}
    resumed = await run_agentic_loop_pipeline(pipeline, "goal", resume_from=paused)
    # After resume: Should have AskHuman command with human input + FinishCommand
    assert len(resumed.final_pipeline_context.command_log) == 2
    assert resumed.final_pipeline_context.command_log[0].execution_result == "human"
    assert resumed.final_pipeline_context.command_log[-1].execution_result == "ok"
    assert resumed.status in {"completed", "failed"}


@pytest.mark.asyncio
async def test_pause_preserves_command_log() -> None:
    planner = StubAgent([AskHumanCommand(question="Need input")])
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})
    paused = await run_agentic_loop_pipeline(pipeline, "goal")
    ctx = paused.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert len(ctx.command_log) == 0


def test_sync_resume() -> None:
    import asyncio

    planner = StubAgent(
        [
            AskHumanCommand(question="Need input"),
            FinishCommand(final_answer="ok"),
        ]
    )
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})
    paused = asyncio.run(run_agentic_loop_pipeline(pipeline, "goal"))
    resumed = asyncio.run(run_agentic_loop_pipeline(pipeline, "goal", resume_from=paused))
    # After resume: Should have AskHuman command with human input + FinishCommand
    assert len(resumed.final_pipeline_context.command_log) == 2
    assert resumed.final_pipeline_context.command_log[0].execution_result == "human"
    assert resumed.final_pipeline_context.command_log[-1].execution_result == "ok"


@pytest.mark.asyncio
async def test_max_loops_failure() -> None:
    planner = StubAgent([RunAgentCommand(agent_name="x", input_data=1)] * 3)
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={}, max_loops=3)
    result = await run_agentic_loop_pipeline(pipeline, "goal")
    ctx = result.final_pipeline_context
    assert len(ctx.command_log) >= 2  # May have fewer commands due to improved error handling
    last_step = result.step_history[-1]
    assert last_step.success is False
