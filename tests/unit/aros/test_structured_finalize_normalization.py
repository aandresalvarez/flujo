from __future__ import annotations

import pytest

from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.application.runner import Flujo


class FencedJsonAgent:
    async def run(self, data, **kwargs):  # type: ignore[no-untyped-def]
        return """
```json
{"value": "ok"}
```
""".strip()


@pytest.mark.anyio
async def test_finalize_parses_fenced_json_to_object() -> None:
    # Build a Step with structured_output intent
    def noop_func(x: str) -> dict[str, str]:
        return {"value": "ok"}

    s = Step.from_callable(noop_func, name="noop")
    # Replace agent with our fenced-json agent
    s.agent = FencedJsonAgent()
    # Declare structured intent in meta to trigger normalization
    s.meta = {"processing": {"structured_output": "openai_json"}}

    # Second step just echoes the previous output
    def echo_func(x: dict[str, str]) -> dict[str, str]:
        return x

    echo = Step.from_callable(echo_func, name="echo")

    p = Pipeline.model_construct(steps=[s, echo])

    runner = Flujo(pipeline=p, pipeline_name="normalize_test")

    async def _run():
        last = None
        async for res in runner.run_async(""):
            last = res
        return last

    result = await _run()
    # Expect both steps recorded
    assert [st.name for st in result.step_history] == ["noop", "echo"]
    # The first step output should be a dict parsed from fenced JSON
    assert isinstance(result.step_history[0].output, dict)
    assert result.step_history[0].output.get("value") == "ok"
