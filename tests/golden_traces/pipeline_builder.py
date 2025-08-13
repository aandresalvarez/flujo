from __future__ import annotations

from typing import Any

from flujo import Step
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.testing.utils import StubAgent


def make_golden_pipeline() -> Step[Any, Any]:
    """Build a comprehensive pipeline used by golden trace tests.

    Contents:
    - Simple step `analyze`
    - Parallel branch with two leaves (p_a, p_b)
    - Loop step that exits after first iteration
    - Primary step with fallback to trigger fallback tracing
    - HITL step to enable pause/resume event coverage in end-to-end flows
    """

    # Simple steps
    analyze = Step.model_validate({"name": "analyze", "agent": StubAgent(["ok"])})
    primary = Step.model_validate(
        {
            "name": "primary",
            "agent": StubAgent([Exception("boom"), "ok"]),
        }
    )
    fb = Step.model_validate({"name": "fallback", "agent": StubAgent(["fb_ok"])})
    primary.fallback(fb)

    # Parallel branch to exercise parallel tracing
    p_a = Step.model_validate({"name": "p_a", "agent": StubAgent(["ok"])})
    p_b = Step.model_validate({"name": "p_b", "agent": StubAgent(["ok"])})
    parallel = ParallelStep(name="par", branches={"a": analyze >> p_a, "b": p_b})

    # Loop to exercise loop tracing (single iteration)
    loop_body = Step.model_validate({"name": "body", "agent": StubAgent(["x", "y"])})
    loop_pipeline = Pipeline.from_step(loop_body)
    loop = LoopStep(
        name="loop",
        loop_body_pipeline=loop_pipeline,
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )

    # Human-in-the-loop step to cover pause/resume in full scenarios
    hitl = HumanInTheLoopStep(name="hitl", message_for_user="confirm")

    # Compose final pipeline
    return parallel >> loop >> primary >> hitl
