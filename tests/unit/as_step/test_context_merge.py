from __future__ import annotations

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("lenient", ["1", "0"])  # exercise both env modes
async def test_as_step_merges_command_log(monkeypatch: pytest.MonkeyPatch, lenient: str) -> None:
    """Ensure runner.as_step merges command_log from the inner run under both lenient modes.

    This test uses the agentic loop factory to generate command_log entries deterministically
    (RunAgent + Finish). It then composes the inner runner via as_step and validates that
    the outer final context contains the last command's execution result.
    """
    from flujo.recipes.factories import make_agentic_loop_pipeline
    from flujo.testing.utils import StubAgent, gather_result
    from flujo.domain.commands import FinishCommand, RunAgentCommand
    from flujo.domain.models import PipelineContext
    from tests.conftest import create_test_flujo

    # Gate the lenient fast-path
    monkeypatch.setenv("FLUJO_LENIENT_AS_STEP_CONTEXT", lenient)

    planner = StubAgent(
        [RunAgentCommand(agent_name="tool", input_data="hi"), FinishCommand(final_answer="done")]
    )
    tool = StubAgent(["tool-output"])
    pipeline = make_agentic_loop_pipeline(planner_agent=planner, agent_registry={"tool": tool})

    inner_runner = create_test_flujo(pipeline, context_model=PipelineContext)
    pipeline_step = inner_runner.as_step(name="loop")
    outer_runner = create_test_flujo(pipeline_step, context_model=PipelineContext)

    res = await gather_result(outer_runner, "goal", initial_context_data={"initial_prompt": "goal"})
    ctx = res.final_pipeline_context
    assert isinstance(ctx.command_log, list)
    assert len(ctx.command_log) >= 1
    assert getattr(ctx.command_log[-1], "execution_result", None) == "done"
