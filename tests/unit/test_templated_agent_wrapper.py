from __future__ import annotations

from typing import Any

import pytest

from flujo.agents.wrapper import TemplatedAsyncAgentWrapper


class _FakeAgent:
    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.called_with_prompt: str | None = None

    async def run(self, x: Any, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - trivial
        # Record the prompt value at call-time
        self.called_with_prompt = self.system_prompt
        return "ok"


@pytest.mark.asyncio
async def test_templated_wrapper_renders_and_restores_prompt() -> None:
    agent = _FakeAgent()
    template = "Hello {{ user }} about {{ previous_step.topic }}"
    variables = {"user": "{{ context.user }}"}
    wrapper = TemplatedAsyncAgentWrapper(
        agent,
        template_string=template,
        variables_spec=variables,
        max_retries=1,
        timeout=5,
        model_name="openai:gpt-4o-mini",
    )

    prev_output = {"topic": "testing"}
    ctx = {"user": "Alice"}

    result = await wrapper.run_async(prev_output, context=ctx)
    assert result == "ok"
    # Ensure the underlying agent saw the rendered prompt
    assert agent.called_with_prompt == "Hello Alice about testing"
    # Ensure original prompt is restored after call
    assert agent.system_prompt is None
