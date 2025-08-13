from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from flujo import Step
from flujo.domain.dsl.parallel import ParallelStep
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.step import HumanInTheLoopStep
from flujo.testing.utils import StubAgent
from flujo.application.runner import Flujo
from tests.golden_traces.utils import span_to_contract_dict, trees_equal


def _make_complex_pipeline() -> Step[Any, Any]:
    # Simple steps
    analyze = Step.model_validate({"name": "analyze", "agent": StubAgent(["ok"])})
    primary = Step.model_validate(
        {"name": "primary", "agent": StubAgent([Exception("boom"), "ok"])}
    )
    fb = Step.model_validate({"name": "fallback", "agent": StubAgent(["fb_ok"])})
    primary.fallback(fb)

    # Parallel branch to exercise parallel tracing
    p_a = Step.model_validate({"name": "p_a", "agent": StubAgent(["ok"])})
    p_b = Step.model_validate({"name": "p_b", "agent": StubAgent(["ok"])})
    parallel = ParallelStep(name="par", branches={"a": analyze >> p_a, "b": p_b})

    # Loop to exercise loop tracing
    loop_body = Step.model_validate({"name": "body", "agent": StubAgent(["x", "y"])})
    from flujo.domain.dsl.pipeline import Pipeline

    loop_pipeline = Pipeline.from_step(loop_body)
    # Minimal loop with exit condition that stops after first iteration
    loop = LoopStep(
        name="loop",
        loop_body_pipeline=loop_pipeline,
        exit_condition_callable=lambda _o, _c: True,
        max_loops=1,
    )

    # Human-in-the-loop to trigger paused/resumed events on resume path (resume is validated elsewhere)
    hitl = HumanInTheLoopStep(name="hitl", message_for_user="confirm")

    # Compose: parallel >> loop >> primary (with fallback) >> hitl (which will pause)
    return parallel >> loop >> primary >> hitl


@pytest.mark.asyncio
async def test_pipeline_trace_matches_golden_file(tmp_path: Path) -> None:
    golden_path = Path(__file__).with_name("golden_trace_v1.json")
    pipe = _make_complex_pipeline()
    # Use in-memory backend by default; disable state persistence to avoid serializing Exception
    runner = Flujo(pipe)
    runner.state_backend = None
    final = None
    async for res in runner.run_async({"x": 1}, run_id=None):
        final = res
    assert final is not None and final.trace_tree is not None
    actual = span_to_contract_dict(final.trace_tree)

    if not golden_path.exists():
        # First run: generate golden
        golden_path.write_text(json.dumps(actual, indent=2))
        pytest.skip("Golden trace generated; re-run to compare.")

    expected = json.loads(golden_path.read_text())
    assert trees_equal(actual, expected)
