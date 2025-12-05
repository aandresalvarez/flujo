from __future__ import annotations

import pytest
from flujo.agents.wrapper import make_agent_async

# Integration tests for GPT-4o-mini model handling.
pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_policy_openai_gpt4o_mini_auto_grammar_applied(monkeypatch):
    from flujo.application.core.executor_core import ExecutorCore
    from flujo.application.core.executor_helpers import make_execution_frame
    from flujo.application.core.step_policies import DefaultAgentStepExecutor
    from flujo.domain.dsl.step import Step
    from flujo.domain.models import Success
    from flujo.tracing.manager import TraceManager, set_active_trace_manager

    wrapper = make_agent_async(
        model="openai:gpt-4o-mini",
        system_prompt="You are a helpful agent",
        output_type=str,
    )

    async def fake_run(payload, **kwargs):  # type: ignore[no-untyped-def]
        return {"output": {"ok": True}}

    monkeypatch.setattr(wrapper._agent, "run", fake_run, raising=True)  # type: ignore[attr-defined]
    step = Step(name="s1", agent=wrapper)
    step.meta["processing"] = {"structured_output": "auto", "schema": {"type": "object"}}

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    core = ExecutorCore()
    frame = make_execution_frame(
        core,
        step,
        "hello",
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

    outcome = await execu.execute(core=core, frame=frame)

    assert isinstance(outcome, Success)
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1
    assert cur.attributes.get("aros.soe.mode") == "openai_json"
