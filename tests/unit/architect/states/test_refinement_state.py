from __future__ import annotations

import pytest

from flujo.architect.context import ArchitectContext
from flujo.architect.states.refinement import build_refinement_state


@pytest.mark.asyncio
async def test_refinement_state_collects_feedback_and_loops() -> None:
    pipeline = build_refinement_state()
    capture_step, goto_step = pipeline.steps

    ctx = ArchitectContext(refinement_feedback="")

    feedback = await capture_step.agent.run(None, context=ctx)
    transition = await goto_step.agent.run(feedback, context=ctx)

    assert "refinement_feedback" in feedback
    assert feedback["refinement_feedback"]
    assert transition["next_state"] == "Planning"
