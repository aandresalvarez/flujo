import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepOutcome, Success


class _NoopAgent:
    async def run(self, *_args, **_kwargs):
        return {"ok": True}


@pytest.mark.asyncio
async def test_simple_executor_path_returns_outcome_in_backend_path():
    # Force routing through SimpleStep policy by providing a fallback_step
    primary = Step(name="primary", agent=_NoopAgent())
    fallback = Step(name="fallback", agent=_NoopAgent())
    step = Step(name="with_fallback", agent=primary.agent, fallback_step=fallback)

    core = ExecutorCore()
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
    assert isinstance(outcome, Success)
    assert outcome.step_result.success is True
    assert outcome.step_result.output == {"ok": True}
