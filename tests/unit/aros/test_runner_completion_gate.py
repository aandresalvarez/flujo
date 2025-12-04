from __future__ import annotations

import pytest

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.application.runner import Flujo


class NoopAgent:
    async def run(self, data, **kwargs):  # type: ignore[no-untyped-def]
        return data


class BadJsonAgent:
    async def run(self, data, **kwargs):  # type: ignore[no-untyped-def]
        # Return a string that is not valid JSON and not parseable
        return "{bad json"


@pytest.mark.anyio
async def test_runner_does_not_mark_completed_with_missing_step_results() -> None:
    # First step succeeds
    def s1_func(x: str) -> str:
        return x

    s1 = Step.from_callable(s1_func, name="s1")
    s1.agent = NoopAgent()

    # Second step declares structured output but returns irreparable JSON
    def s2_func(x: str) -> str:
        return x

    s2 = Step.from_callable(s2_func, name="s2")
    s2.agent = BadJsonAgent()
    s2.meta = {"processing": {"structured_output": "openai_json"}}

    p = Pipeline.model_construct(steps=[s1, s2])
    runner = Flujo(pipeline=p, pipeline_name="gate_test")

    async def _run():
        last = None
        async for res in runner.run_async(""):
            last = res
        return last

    result = await _run()
    # With the completion gate, we should not mark completed unless both succeeded
    # Expect either failed status (result.success False) or a non-two-length history
    # Here we assert success flag is False when fewer than two successful steps were recorded
    if len(result.step_history) != 2 or not all(st.success for st in result.step_history):
        assert result.success is False
