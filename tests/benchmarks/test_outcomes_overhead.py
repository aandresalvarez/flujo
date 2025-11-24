import time

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.types import ExecutionFrame
from flujo.domain.dsl.step import Step
from flujo.domain.models import StepResult


class _NoopAgent:
    async def run(self, *_args, **_kwargs):
        return {"ok": True}


@pytest.mark.asyncio
async def test_outcomes_adapter_overhead_smoke():
    core = ExecutorCore()
    step = Step(name="noop", agent=_NoopAgent())

    # Legacy path (no frame)
    t0 = time.perf_counter()
    r1 = await core.execute(step, None)
    t1 = time.perf_counter() - t0

    assert isinstance(r1, (StepResult,))

    # Backend/frame path (uses outcomes adapter)
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
    t2 = time.perf_counter()
    await core.execute(frame)
    t3 = time.perf_counter() - t2

    # Basic sanity: the adapter path shouldn't be orders of magnitude slower
    # This is a smoke check; precise thresholds belong in dedicated perf runs
    assert t3 < (t1 * 10 + 0.050)
