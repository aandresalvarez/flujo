import pytest

from flujo import Flujo, Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.telemetry import OpenTelemetryHook


@pytest.mark.asyncio
async def test_pipeline_runs_with_otel_hook(tmp_path):
    hook = OpenTelemetryHook(mode="console")
    step = Step.model_validate({"name": "s", "agent": StubAgent(["o"])})
    runner = Flujo(step, hooks=[hook.hook], state_backend=None)
    result = await gather_result(runner, "in")
    assert result.step_history[-1].success
