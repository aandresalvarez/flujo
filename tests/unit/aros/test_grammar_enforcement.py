from __future__ import annotations

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success, Failure
from flujo.tracing.manager import TraceManager, set_active_trace_manager


@pytest.mark.fast
@pytest.mark.asyncio
async def test_outlines_enforcement_fails_on_non_matching_output():
    class _Agent:
        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            return "not_json"

    step = Step(name="s1", agent=_Agent())
    step.meta["processing"] = {
        "structured_output": "outlines",
        "enforce_grammar": True,
        "schema": {"type": "object"},
    }

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        breach_event=None,
        _fallback_depth=0,
    )
    assert isinstance(outcome, Failure)
    assert "did not match enforced grammar" in (outcome.step_result.feedback or "")


@pytest.mark.fast
@pytest.mark.asyncio
async def test_outlines_enforcement_passes_on_matching_output():
    class _Agent:
        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            return "{}"

    step = Step(name="s1", agent=_Agent())
    step.meta["processing"] = {
        "structured_output": "outlines",
        "enforce_grammar": True,
        "schema": {"type": "object"},
    }

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        breach_event=None,
        _fallback_depth=0,
    )
    assert isinstance(outcome, Success)
