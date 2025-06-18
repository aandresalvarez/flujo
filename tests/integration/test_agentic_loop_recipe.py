from unittest.mock import AsyncMock
import pytest

from flujo.recipes.agentic_loop import AgenticLoop
from flujo.domain.commands import (
    RunAgentCommand,
    RunPythonCodeCommand,
    AskHumanCommand,
    FinishCommand,
)
from flujo.testing.utils import StubAgent
from flujo.domain.models import PipelineContext


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
    loop = AgenticLoop(planner, {"summarizer": summarizer})
    result = await loop.run_async("goal")
    summarizer.run.assert_called_once()
    args, kwargs = summarizer.run.call_args
    assert args[0] == "hi"
    ctx = result.final_pipeline_context
    assert isinstance(ctx, PipelineContext)
    assert len(ctx.command_log) == 2
    assert ctx.command_log[-1].execution_result == "done"


@pytest.mark.asyncio
async def test_pause_and_resume_in_loop() -> None:
    planner = StubAgent(
        [
            AskHumanCommand(question="Need input"),
            FinishCommand(final_answer="ok"),
        ]
    )
    loop = AgenticLoop(planner, {})
    paused = await loop.run_async("goal")
    ctx = paused.final_pipeline_context
    assert ctx.scratchpad["status"] == "paused"
    resumed = await loop.resume_async(paused, "human")
    assert resumed.final_pipeline_context.command_log[0].execution_result == "human"
    assert resumed.final_pipeline_context.scratchpad["status"] == "completed"


def test_sync_resume() -> None:
    planner = StubAgent(
        [
            AskHumanCommand(question="Need input"),
            FinishCommand(final_answer="ok"),
        ]
    )
    loop = AgenticLoop(planner, {})
    paused = loop.run("goal")
    resumed = loop.resume(paused, "human")
    assert resumed.final_pipeline_context.command_log[0].execution_result == "human"


@pytest.mark.asyncio
async def test_max_loops_failure() -> None:
    planner = StubAgent([RunAgentCommand(agent_name="x", input_data=1)])
    loop = AgenticLoop(planner, {}, max_loops=3)
    result = await loop.run_async("goal")
    ctx = result.final_pipeline_context
    assert len(ctx.command_log) == 3
    last_step = result.step_history[-1]
    assert last_step.success is False


@pytest.mark.asyncio
async def test_run_python_safe() -> None:
    planner = StubAgent(
        [
            RunPythonCodeCommand(code="result = 1 + 1"),
            FinishCommand(final_answer="done"),
        ]
    )
    loop = AgenticLoop(planner, {})
    result = await loop.run_async("goal")
    ctx = result.final_pipeline_context
    assert ctx.command_log[0].execution_result == 2


@pytest.mark.asyncio
async def test_run_python_rejects_imports() -> None:
    planner = StubAgent(
        [
            RunPythonCodeCommand(code="import os\nresult = 42"),
            FinishCommand(final_answer="done"),
        ]
    )
    loop = AgenticLoop(planner, {})
    result = await loop.run_async("goal")
    log_entry = result.final_pipeline_context.command_log[0]
    assert "Imports are not allowed" in str(log_entry.execution_result)
