import pytest
from unittest.mock import MagicMock
from flujo.infra.console_tracer import ConsoleTracer
from flujo.domain.models import StepResult
from flujo.domain.events import PostStepPayload


@pytest.mark.asyncio
async def test_console_tracer_hook_prints() -> None:
    tracer = ConsoleTracer(level="debug")
    spy = MagicMock()
    setattr(tracer.console, "print", spy)
    result = StepResult(name="s", output="out")
    payload = PostStepPayload(event_name="post_step", step_result=result)
    await tracer.hook(payload)
    spy.assert_called()


def test_console_tracer_config() -> None:
    tracer = ConsoleTracer(level="info", log_inputs=False, log_outputs=False)
    assert tracer.level == "info"
    assert not tracer.log_inputs
    assert not tracer.log_outputs
