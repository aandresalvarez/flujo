from __future__ import annotations

import pytest
from flujo.agents import make_agent
from flujo.agents.wrapper import make_agent_async, AsyncAgentWrapper

# Integration tests for GPT-5 family model handling and structured output.
pytestmark = pytest.mark.slow


def test_make_agent_uses_openai_responses_model_for_gpt5_family():
    try:
        from pydantic_ai.models.openai import OpenAIResponsesModel
    except Exception:  # pragma: no cover - environment may not have this model
        pytest.skip("pydantic-ai OpenAIResponsesModel not available")

    # Create agent for GPT-5 family; should construct an OpenAIResponsesModel
    agent, _ = make_agent(
        model="openai:gpt-5-mini",
        system_prompt="You are a helpful agent",
        output_type=str,
    )
    assert isinstance(agent.model, OpenAIResponsesModel)


@pytest.mark.asyncio
async def test_make_agent_async_allows_structured_output_hint_without_network(monkeypatch):
    # Build a GPT-5 family agent wrapper
    wrapper = make_agent_async(
        model="openai:gpt-5-mini",
        system_prompt="You are a helpful agent",
        output_type=str,
    )
    assert isinstance(wrapper, AsyncAgentWrapper)

    # Monkeypatch underlying pydantic-ai Agent.run to avoid network and capture kwargs
    captured: dict | None = None

    async def fake_run(payload, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured
        captured = dict(kwargs)
        return {"output": {"ok": True}}

    # Assign coroutine directly
    monkeypatch.setattr(wrapper._agent, "run", fake_run, raising=True)  # type: ignore[attr-defined]

    # Enable structured output and run
    wrapper.enable_structured_output(json_schema={"type": "object"}, name="gpt5")
    await wrapper.run_async("hello")
    assert isinstance(captured, dict)
    rf = captured.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") in {"json_schema", "json_object"}


@pytest.mark.asyncio
async def test_make_agent_async_gpt4o_structured_output_hint_without_network(monkeypatch):
    wrapper = make_agent_async(
        model="openai:gpt-4o",
        system_prompt="You are a helpful agent",
        output_type=str,
    )
    captured: dict | None = None

    async def fake_run(payload, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured
        captured = dict(kwargs)
        return {"output": {"ok": True}}

    monkeypatch.setattr(wrapper._agent, "run", fake_run, raising=True)  # type: ignore[attr-defined]
    wrapper.enable_structured_output(json_schema={"type": "object"}, name="gpt4o")
    await wrapper.run_async("hello")
    assert isinstance(captured, dict)
    rf = captured.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") in {"json_schema", "json_object"}


@pytest.mark.asyncio
async def test_policy_openai_gpt4o_auto_grammar_applied(monkeypatch):
    from flujo.application.core.executor_core import ExecutorCore
    from flujo.application.core.executor_helpers import make_execution_frame
    from flujo.application.core.step_policies import DefaultAgentStepExecutor
    from flujo.domain.dsl.step import Step
    from flujo.domain.models import Success
    from flujo.tracing.manager import TraceManager, set_active_trace_manager

    wrapper = make_agent_async(
        model="openai:gpt-4o",
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


@pytest.mark.asyncio
async def test_policy_enables_structured_output_for_gpt5_family_without_network(monkeypatch):
    from flujo.application.core.executor_core import ExecutorCore
    from flujo.application.core.executor_helpers import make_execution_frame
    from flujo.application.core.step_policies import DefaultAgentStepExecutor
    from flujo.domain.dsl.step import Step
    from flujo.domain.models import Success
    from flujo.tracing.manager import TraceManager, set_active_trace_manager

    # Build wrapper and monkeypatch run
    wrapper = make_agent_async(
        model="openai:gpt-5-mini",
        system_prompt="You are a helpful agent",
        output_type=str,
    )

    async def fake_run(payload, **kwargs):  # type: ignore[no-untyped-def]
        return {"output": {"ok": True}}

    monkeypatch.setattr(wrapper._agent, "run", fake_run, raising=True)  # type: ignore[attr-defined]

    # Create step with structured_output auto
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
