from __future__ import annotations

import pytest

from pydantic import BaseModel

from flujo.agents.wrapper import AsyncAgentWrapper


class Model(BaseModel):
    value: int


class RaisesUnexpected:
    output_type = Model

    async def run(self, *_a, **_k):  # type: ignore[no-untyped-def]
        # Simulate provider JSON-mode failure surfaced by pydantic-ai
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        raise UnexpectedModelBehavior('{"value": 7} trailing text')


@pytest.mark.fast
@pytest.mark.asyncio
async def test_wrapper_repairs_unexpected_model_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    # Route exception text through deterministic repair
    from flujo.agents import utils as agents_utils

    monkeypatch.setattr(
        agents_utils,
        "get_raw_output_from_exception",
        lambda _exc: '{"value": 7} trailing text',
    )

    # Avoid invoking real LLM repair agent
    from flujo.agents import repair as repair_mod

    class DummyRepairAgent:
        async def run(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return '{"value": 7}'

    monkeypatch.setattr(repair_mod, "get_repair_agent", lambda: DummyRepairAgent())

    from flujo.domain.agent_result import FlujoAgentResult

    wrapper = AsyncAgentWrapper(RaisesUnexpected(), max_retries=1, auto_repair=True)
    out = await wrapper.run_async("irrelevant")
    # Wrapper now returns FlujoAgentResult; access the repaired output inside
    assert isinstance(out, FlujoAgentResult)
    assert out.output.value == 7
