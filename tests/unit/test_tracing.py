import pytest
from unittest.mock import MagicMock
from flujo.tracing import ConsoleTracer
from flujo.domain.models import StepResult


@pytest.mark.asyncio  # type: ignore[misc]
async def test_console_tracer_hook_prints() -> None:
    tracer = ConsoleTracer(level="debug")
    spy = MagicMock()
    tracer.console.print = spy
    result = StepResult(name="s", output="out")
    await tracer.hook(event_name="post_step", step_result=result)
    spy.assert_called()


def test_console_tracer_config() -> None:
    tracer = ConsoleTracer(level="info", log_inputs=False, log_outputs=False)
    assert tracer.level == "info"
    assert not tracer.log_inputs
    assert not tracer.log_outputs
