import pytest
from unittest.mock import MagicMock
from typing import Any, cast

from flujo import Step
from flujo.domain.agent_protocol import AsyncAgentProtocol
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.events import HookPayload, PostStepPayload
from tests.conftest import create_test_flujo


@pytest.mark.asyncio
async def test_hook_receives_typed_payload() -> None:
    step = Step.model_validate(
        {"name": "s1", "agent": cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))}
    )
    recorder = MagicMock()

    async def hook(payload: HookPayload) -> None:
        recorder(payload)

    runner = create_test_flujo(step, hooks=[hook])
    await gather_result(runner, "start")

    post_step_calls = [
        c.args[0] for c in recorder.call_args_list if isinstance(c.args[0], PostStepPayload)
    ]
    assert len(post_step_calls) == 1
    payload = post_step_calls[0]
    assert isinstance(payload, PostStepPayload)
    assert payload.step_result.output == "ok"
