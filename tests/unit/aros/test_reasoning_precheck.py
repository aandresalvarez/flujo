from __future__ import annotations

import pytest

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.executor_helpers import make_execution_frame
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.tracing.manager import TraceManager, set_active_trace_manager


class _Agent:
    async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        return "ok"


class _Validator:
    async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("validator should not be called when no plan")


@pytest.mark.fast
@pytest.mark.asyncio
async def test_reasoning_precheck_skipped_when_no_plan_found():
    step = Step(name="s1", agent=_Agent())
    step.meta["processing"] = {
        "reasoning_precheck": {
            "enabled": True,
            "validator_agent": _Validator(),
            "delimiters": ["<thinking>", "</thinking>"],
        }
    }

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    # Data without plan delimiters
    frame = make_execution_frame(
        core,
        step,
        "no markers here",
        context=None,
        resources=None,
        limits=None,
        context_setter=None,
        stream=False,
        on_chunk=None,
        fallback_depth=0,
        result=None,
        quota=None,
    )
    _ = await execu.execute(core=core, frame=frame)
    cur = tm._span_stack[-1]
    # Ensure skipped event was recorded
    found = any(ev.get("name") == "aros.reasoning.precheck.skipped" for ev in cur.events)
    assert found
