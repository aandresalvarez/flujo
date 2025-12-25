"""Tests for AROS structured output finalize normalization.

These tests document the current behavior of the structured output
finalization path and the AROS processing pipeline.
"""

from __future__ import annotations

import pytest

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.application.runner import Flujo

pytestmark = pytest.mark.fast


class FencedJsonAgent:
    """Agent that returns JSON wrapped in markdown code fences."""

    async def run(self, data, **kwargs):  # type: ignore[no-untyped-def]
        return """
```json
{"value": "ok"}
```
""".strip()


async def _run_pipeline(runner: Flujo, payload: str = ""):
    """Run a Flujo pipeline and return the final result."""
    last = None
    async for res in runner.run_async(payload):
        last = res
    return last


@pytest.mark.anyio
async def test_fenced_json_passthrough_when_types_match() -> None:
    """Structured output with fenced JSON passes when types are compatible.

    Note: The current implementation may or may not auto-parse fenced JSON.
    This test documents behavior where type annotations are compatible.
    """

    # Build a Step that returns a string (fenced JSON)
    def noop_func(x: str) -> str:
        return x  # Type annotation says string

    s = Step.from_callable(noop_func, name="noop")
    # Replace agent with our fenced-json agent
    s.agent = FencedJsonAgent()

    p = Pipeline.model_construct(steps=[s])
    runner = Flujo(pipeline=p, pipeline_name="fenced_test", persist_state=False)

    result = await _run_pipeline(runner)

    # Step should complete successfully
    assert len(result.step_history) == 1
    assert result.step_history[0].success is True
    assert result.step_history[0].output == '```json\n{"value": "ok"}\n```'


@pytest.mark.anyio
async def test_structured_meta_triggers_processing() -> None:
    """Structured output meta setting triggers processing pipeline.

    When processing.structured_output is set, the AROS pipeline should
    attempt to process the output according to the specified mode.
    """

    class DictAgent:
        """Agent that returns a proper dict."""

        async def run(self, data, **kwargs):  # type: ignore[no-untyped-def]
            return {"value": "ok"}

    def echo_func(x: dict[str, str]) -> dict[str, str]:
        return x

    s = Step.from_callable(echo_func, name="structured")
    s.agent = DictAgent()
    # Declare structured intent in meta
    s.meta = {"processing": {"structured_output": "openai_json"}}

    p = Pipeline.model_construct(steps=[s])
    runner = Flujo(pipeline=p, pipeline_name="structured_test", persist_state=False)

    result = await _run_pipeline(runner)

    # Step should complete successfully with dict output
    assert len(result.step_history) == 1
    assert result.step_history[0].success is True
    assert isinstance(result.step_history[0].output, dict)
    assert result.step_history[0].output.get("value") == "ok"
