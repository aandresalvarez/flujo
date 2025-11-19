"""
Regression tests for pause propagation through nested control-flow policies.

These tests ensure ConditionalStep (inside LoopStep) and ParallelStep bubble
PausedException all the way to the runner so nested HITL steps behave correctly.
"""

from __future__ import annotations

import pytest

from flujo import Flujo
from flujo.domain.dsl import Pipeline, Step, LoopStep, HumanInTheLoopStep
from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.models import PipelineResult
from flujo.testing.utils import StubAgent


pytestmark = [pytest.mark.slow, pytest.mark.serial]


async def _run_until_pause(runner: Flujo, payload: object) -> PipelineResult:
    paused_result = None
    async for result in runner.run_async(payload):
        paused_result = result
        break
    assert paused_result is not None, "Runner should yield a PipelineResult before exiting"
    return paused_result


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_loop_conditional_hitl_propagates_pause() -> None:
    """Loop -> Conditional -> HITL pauses the entire pipeline."""
    hitl_step = HumanInTheLoopStep(
        name="nested_hitl",
        message_for_user="Provide additional input",
    )
    hitl_branch = Pipeline.from_step(hitl_step)
    conditional = ConditionalStep(
        name="route_to_hitl",
        condition_callable=lambda _out, _ctx: "hitl",
        branches={"hitl": hitl_branch},
    )
    loop_body = Pipeline.from_step(conditional)
    loop = LoopStep(
        name="loop_with_conditional",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda _output, _context: True,
        max_loops=1,
    )
    pipeline = Pipeline.from_step(loop)
    runner = Flujo(pipeline)

    paused_result = await _run_until_pause(runner, {"input": "start"})
    ctx = paused_result.final_pipeline_context
    assert ctx is not None and hasattr(ctx, "scratchpad"), "Loop context should exist"
    assert ctx.scratchpad.get("status") == "paused", "Runner must surface a paused state"


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_parallel_hitl_branch_propagates_pause() -> None:
    """Parallel branch hitting HITL pauses the entire runner."""
    hitl_step = HumanInTheLoopStep(
        name="parallel_hitl",
        message_for_user="Confirm parallel branch",
    )
    agent_step = Step(name="agent_branch", agent=StubAgent(["done"]))
    parallel = ParallelStep(
        name="parallel_gate",
        branches={
            "hitl": Pipeline.from_step(hitl_step),
            "agent": Pipeline.from_step(agent_step),
        },
    )
    pipeline = Pipeline.from_step(parallel)
    runner = Flujo(pipeline)

    paused_result = await _run_until_pause(runner, "payload")
    ctx = paused_result.final_pipeline_context
    assert ctx is not None and hasattr(ctx, "scratchpad")
    assert ctx.scratchpad.get("status") == "paused", "Parallel HITL should pause the runner"
