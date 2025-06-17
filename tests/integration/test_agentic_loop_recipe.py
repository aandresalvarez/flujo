from unittest.mock import AsyncMock
import pytest

from flujo.recipes.agentic_loop import AgenticLoop
from flujo.domain.commands import (
    RunAgentCommand,
    AskHumanCommand,
    FinishCommand,
)
from flujo.testing.utils import StubAgent
from flujo.domain.models import PipelineContext


@pytest.mark.asyncio
async def test_agent_delegation_and_finish() -> None:
    planner = StubAgent([
        RunAgentCommand(agent_name="summarizer", input_data="hi"),
        FinishCommand(final_answer="done"),
    ])
    summarizer = AsyncMock()
    summarizer.run = AsyncMock(return_value="summary")
    loop = AgenticLoop(planner, {"summarizer": summarizer})
    result = await loop.run_async("goal")
    summarizer.run.assert_called_once_with("hi")
    ctx = result.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert len(ctx.command_log) == 2
    assert ctx.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_pause_and_resume_in_loop() -> None:
    planner = StubAgent([
        AskHumanCommand(question="Need input"),
        FinishCommand(final_answer="ok"),
    ])
    loop = AgenticLoop(planner, {})
    paused = await loop.run_async("goal")
    ctx = paused.final_pipeline_context
    assert ctx.scratchpad["status"] == "paused"
    resumed = await loop.resume_async(paused, "human")
    assert resumed.final_pipeline_context.command_log[0].execution_result == "human"
    assert resumed.final_pipeline_context.scratchpad["status"] == "completed"


@pytest.mark.asyncio
async def test_max_loops_failure() -> None:
    planner = StubAgent([RunAgentCommand(agent_name="x", input_data=1)])
    loop = AgenticLoop(planner, {}, max_loops=3)
    result = await loop.run_async("goal")
    ctx = result.final_pipeline_context
    assert len(ctx.command_log) == 3
    last_step = result.step_history[-1]
    assert last_step.success is False
