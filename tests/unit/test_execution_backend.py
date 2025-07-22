import pytest

from flujo import Step
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.backends import StepExecutionRequest
from flujo.domain.models import StepResult
from tests.conftest import create_test_flujo


class DummyBackend:
    def __init__(self):
        self.called = 0
        self.agent_registry = {}

    async def execute_step(self, request: StepExecutionRequest) -> StepResult:
        self.called += 1
        return StepResult(name=request.step.name, output="ok")


@pytest.mark.asyncio
async def test_custom_backend_invoked() -> None:
    backend = DummyBackend()
    step = Step.model_validate({"name": "s", "agent": StubAgent(["ignored"])})
    runner = create_test_flujo(step, backend=backend)
    result = await gather_result(runner, "in")
    assert backend.called == 1
    assert result.step_history[0].output == "ok"
    assert step.agent.call_count == 0
