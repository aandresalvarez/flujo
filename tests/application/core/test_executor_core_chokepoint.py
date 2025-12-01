import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.types import ExecutionFrame
from flujo.domain.models import Failure, StepOutcome
from tests.test_types.fixtures import create_test_step


class _RaisingAgentExecutor:
    async def execute(self, *args, **kwargs):
        raise ValueError("kaboom")


@pytest.mark.asyncio
async def test_execute_frame_converts_unexpected_exception_into_failure_outcome():
    core = ExecutorCore()
    # Force routing through the Agent policy and make it raise unexpectedly
    core.agent_step_executor = _RaisingAgentExecutor()

    step = create_test_step(name="boom", agent=object())
    frame = ExecutionFrame(
        step=step,
        data=None,
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        context_setter=lambda _r, _c: None,
    )

    outcome = await core.execute(frame)

    assert isinstance(outcome, StepOutcome)
    assert isinstance(outcome, Failure)
    assert outcome.step_result is not None
    assert outcome.step_result.success is False
    # Feedback should include the original error message
    fb = (outcome.feedback or "") + " " + (outcome.step_result.feedback or "")
    assert "kaboom" in fb
