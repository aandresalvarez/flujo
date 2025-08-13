from __future__ import annotations

from typing import Any

import pytest

from flujo import Step
from flujo.testing.utils import StubAgent
from flujo.application.runner import Flujo


def _make_complex_pipeline_unique_for_golden() -> Step[Any, Any]:
    # Simple demo pipeline: analyze >> maybe_fail (with fallback)
    analyze = Step.model_validate({"name": "analyze", "agent": StubAgent(["ok"])})
    primary = Step.model_validate(
        {"name": "primary", "agent": StubAgent([Exception("boom"), "ok"])}
    )
    fb = Step.model_validate({"name": "fallback", "agent": StubAgent(["fb_ok"])})
    primary.fallback(fb)
    return analyze >> primary


@pytest.mark.asyncio
async def test_golden_trace_pipeline_builds_smoke(tmp_path) -> None:
    pipe = _make_complex_pipeline_unique_for_golden()
    runner = Flujo(pipe)
    # Disable persistence to avoid serializing Exception in state
    runner.state_backend = None
    final = None
    async for res in runner.run_async({"x": 1}, run_id=None):
        final = res
    assert final is not None
    assert final.trace_tree is not None
    assert final.trace_tree.name == "pipeline_run"
    assert len(final.trace_tree.children) > 0
