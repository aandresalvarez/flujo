from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from flujo import Step
from flujo.testing.utils import StubAgent
from flujo.application.runner import Flujo
from tests.golden_traces.utils import span_to_contract_dict, trees_equal


def _make_complex_pipeline() -> Step[Any, Any]:
    analyze = Step.model_validate({"name": "analyze", "agent": StubAgent(["ok"])})
    primary = Step.model_validate(
        {"name": "primary", "agent": StubAgent([Exception("boom"), "ok"])}
    )
    fb = Step.model_validate({"name": "fallback", "agent": StubAgent(["fb_ok"])})
    primary.fallback(fb)
    return analyze >> primary


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
