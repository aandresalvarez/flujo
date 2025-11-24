"""Integration coverage for Paused control-flow bubbling in nested structures."""

from __future__ import annotations

import pytest

from flujo.domain.dsl.conditional import ConditionalStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import (
    BranchFailureStrategy,
    HumanInTheLoopStep,
    Step,
)
from flujo.domain.models import Paused
from tests.conftest import create_test_flujo


class _EchoAgent:
    async def run(self, payload, context=None, resources=None, **kwargs):
        return f"ok:{payload}"


@pytest.mark.asyncio
async def test_nested_pause_bubbles_from_loop_conditional_hitl() -> None:
    """Loop → Conditional → HITL should surface a Paused outcome."""
    hitl_step = HumanInTheLoopStep(name="hitl_nested", message_for_user="pause here")
    conditional = ConditionalStep(
        name="route_to_hitl",
        condition_callable=lambda _out, _ctx: "hitl",
        branches={"hitl": Pipeline.from_step(hitl_step)},
    )
    loop = LoopStep(
        name="outer_loop",
        loop_body_pipeline=Pipeline.from_step(conditional),
        exit_condition_callable=lambda _out, _ctx: True,
        max_loops=1,
    )
    runner = create_test_flujo(Pipeline.from_step(loop))

    outcomes = []
    async for item in runner.run_outcomes_async({"payload": "start"}):
        outcomes.append(item)
        break

    assert outcomes and isinstance(outcomes[0], Paused)


@pytest.mark.asyncio
async def test_parallel_branch_hitl_pauses_entire_step() -> None:
    """Parallel → HITL branch should pause the whole parallel step."""
    branches = {
        "fast": Step(name="fast_branch", agent=_EchoAgent()),
        "hitl": Pipeline.from_step(
            HumanInTheLoopStep(name="hitl_branch", message_for_user="pause here too")
        ),
    }
    parallel_step = ParallelStep(
        name="parallel_hitl",
        branches=branches,
        on_branch_failure=BranchFailureStrategy.PROPAGATE,
    )
    runner = create_test_flujo(Pipeline.from_step(parallel_step))

    outcomes = []
    async for item in runner.run_outcomes_async("payload"):
        outcomes.append(item)
        break

    assert outcomes and isinstance(outcomes[0], Paused)
