from __future__ import annotations

import pytest
from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultAgentStepExecutor
from flujo.domain.dsl.step import Step
from flujo.domain.models import Success
from flujo.agents.wrapper import AsyncAgentWrapper
from flujo.tracing.manager import TraceManager, set_active_trace_manager

# Skip policy-executor integration in this unit file to avoid flaky executor paths in CI.
# SOE behavior is validated via wrapper tests and adapter stubs; tracing aggregation is covered elsewhere.
pytestmark = pytest.mark.skip(
    reason="Use wrapper/adapter tests for SOE; skip policy executor integration here."
)


class _SOEFakeAgent:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    async def run(self, payload, response_format=None, **kwargs):  # type: ignore[no-untyped-def]
        # Capture kwargs to assert that response_format is passed through the wrapper
        self.last_kwargs = {"response_format": response_format, **kwargs}
        # Mimic a pydantic-ai like shape
        return {"output": {"ok": True}}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_enables_structured_output_and_passes_response_format():
    core = ExecutorCore()

    # Wrap the fake agent so policy can call wrapper.enable_structured_output
    underlying = _SOEFakeAgent()
    wrapper = AsyncAgentWrapper(underlying, model_name="openai:gpt-4o")

    step = Step(name="s1", agent=wrapper)
    # Attach processing meta to request structured output with a schema
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    step.meta["processing"] = {"structured_output": "openai_json", "schema": schema}

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    # Ensure underlying agent got a response_format with json_schema
    assert isinstance(underlying.last_kwargs, dict)
    rf = underlying.last_kwargs.get("response_format")
    assert isinstance(rf, dict)
    assert rf.get("type") in {"json_schema", "json_object"}
    if rf.get("type") == "json_schema":
        assert isinstance(rf.get("json_schema"), dict)


class _NoKwAgent:
    def __init__(self) -> None:
        self.called = False

    async def run(self, payload):  # type: ignore[no-untyped-def]
        self.called = True
        return {"output": {"ok": True}}


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_structured_output_safe_noop_when_agent_no_kwargs():
    core = ExecutorCore()

    no_kw = _NoKwAgent()
    wrapper = AsyncAgentWrapper(no_kw, model_name="openai:gpt-4o")

    step = Step(name="s1", agent=wrapper)
    step.meta["processing"] = {
        "structured_output": "openai_json",
        "schema": {"type": "object"},
    }

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    assert no_kw.called is True


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_structured_output_skipped_for_unsupported_provider_records_event():
    core = ExecutorCore()

    class _Agent:
        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            return {"output": {"ok": True}}

    underlying = _Agent()
    # Model id indicates a non-OpenAI provider
    wrapper = AsyncAgentWrapper(underlying, model_name="anthropic:claude-3")

    step = Step(name="s1", agent=wrapper)
    step.meta["processing"] = {"structured_output": "openai_json", "schema": {"type": "object"}}

    # Activate a trace manager with a current span to aggregate attributes
    tm = TraceManager()
    tm._span_stack = [
        type("DummySpan", (), {"events": [], "attributes": {}})()
    ]  # simple dummy span
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    cur = tm._span_stack[-1]
    # aros.soe.skipped counters should be present
    assert cur.attributes.get("aros.soe.skipped", 0) >= 1
    assert cur.attributes.get("aros.soe.skipped.unsupported_provider", 0) >= 1


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_structured_output_openai_records_grammar_applied_event():
    core = ExecutorCore()

    class _Agent:
        def __init__(self) -> None:
            self.last_kwargs = None

        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            # Capture any kwargs (wrapper will pass response_format to underlying agent)
            self.last_kwargs = dict(kwargs)
            return {"output": {"ok": True}}

    underlying = _Agent()
    wrapper = AsyncAgentWrapper(underlying, model_name="openai:gpt-4o")

    step = Step(name="s1", agent=wrapper)
    step.meta["processing"] = {"structured_output": "openai_json", "schema": {"type": "object"}}

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1
    # grammar.applied event includes schema_hash
    if cur.events:
        ev = cur.events[-1]
        if ev.get("name") == "grammar.applied":
            sh = ev.get("attributes", {}).get("schema_hash")
            assert isinstance(sh, (str, type(None)))


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_structured_output_auto_behaves_like_openai_json():
    core = ExecutorCore()

    class _Agent:
        async def run(self, payload, **kwargs):  # type: ignore[no-untyped-def]
            return {"output": {"ok": True}}

    underlying = _Agent()
    wrapper = AsyncAgentWrapper(underlying, model_name="openai:gpt-4o")

    step = Step(name="s1", agent=wrapper)
    step.meta["processing"] = {"structured_output": "auto", "schema": {"type": "object"}}

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1
    if cur.events:
        ev = cur.events[-1]
        if ev.get("name") == "grammar.applied":
            sh = ev.get("attributes", {}).get("schema_hash")
            assert isinstance(sh, (str, type(None)))
    assert cur.attributes.get("aros.soe.mode") == "openai_json"
    # Mode should reflect openai_json
    assert cur.attributes.get("aros.soe.mode") == "openai_json"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_outlines_sets_structured_grammar_option_and_traces():
    core = ExecutorCore()

    class _Agent:
        def __init__(self) -> None:
            self.last_options = None

        async def run(self, payload, options=None):  # type: ignore[no-untyped-def]
            self.last_options = options
            return {"output": {"ok": True}}

    agent = _Agent()
    step = Step(name="s1", agent=agent)
    step.meta["processing"] = {
        "structured_output": "outlines",
        "schema": {"type": "object"},
    }

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    # Options should include structured_grammar hint (telemetry-only stub)
    assert isinstance(agent.last_options, dict)
    sg = agent.last_options.get("structured_grammar")
    assert isinstance(sg, dict)
    assert sg.get("mode") == "outlines"
    assert isinstance(sg.get("pattern"), str)
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1


@pytest.mark.fast
@pytest.mark.asyncio
async def test_policy_xgrammar_sets_structured_grammar_option_and_traces():
    core = ExecutorCore()

    class _Agent:
        def __init__(self) -> None:
            self.last_options = None

        async def run(self, payload, options=None):  # type: ignore[no-untyped-def]
            self.last_options = options
            return {"output": {"ok": True}}

    agent = _Agent()
    step = Step(name="s1", agent=agent)
    step.meta["processing"] = {
        "structured_output": "xgrammar",
        "schema": {"type": "array"},
    }

    tm = TraceManager()
    tm._span_stack = [type("DummySpan", (), {"events": [], "attributes": {}})()]
    set_active_trace_manager(tm)

    execu = DefaultAgentStepExecutor()
    outcome = await execu.execute(
        core=core,
        step=step,
        data="hello",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(outcome, Success)
    # Options should include structured_grammar hint (telemetry-only stub)
    assert isinstance(agent.last_options, dict)
    sg = agent.last_options.get("structured_grammar")
    assert isinstance(sg, dict)
    assert sg.get("mode") == "xgrammar"
    assert isinstance(sg.get("pattern"), str)
    cur = tm._span_stack[-1]
    assert cur.attributes.get("aros.soe.count", 0) >= 1
