from unittest.mock import AsyncMock
import pytest

from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.domain.commands import (
    RunAgentCommand,
    AskHumanCommand,
    FinishCommand,
)
from flujo.testing.utils import StubAgent
from flujo.domain.models import PipelineContext


def test_agentic_loop_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning):
        from flujo.recipes.agentic_loop import AgenticLoop

        AgenticLoop(StubAgent([]), {})


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
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={})
    paused = await run_agentic_loop_pipeline(pipeline, "goal")
    ctx = paused.final_pipeline_context
    assert ctx.scratchpad["status"] == "paused"
    resumed = await run_agentic_loop_pipeline(pipeline, "goal", resume_from=paused)
    assert len(resumed.final_pipeline_context.command_log) == 1
    assert resumed.final_pipeline_context.command_log[-1].execution_result == "human"
    assert resumed.final_pipeline_context.scratchpad["status"] == "completed"


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
    assert len(resumed.final_pipeline_context.command_log) == 1
    assert resumed.final_pipeline_context.command_log[-1].execution_result == "human"


@pytest.mark.asyncio
async def test_max_loops_failure() -> None:
    planner = StubAgent([RunAgentCommand(agent_name="x", input_data=1)] * 3)
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={}, max_loops=3)
    result = await run_agentic_loop_pipeline(pipeline, "goal")
    ctx = result.final_pipeline_context
    assert len(ctx.command_log) == 3
    last_step = result.step_history[-1]
    assert last_step.success is False
