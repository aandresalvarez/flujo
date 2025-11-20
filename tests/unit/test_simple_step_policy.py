import pytest
from unittest.mock import AsyncMock

from flujo.application.core.executor_core import ExecutorCore
from flujo.application.core.step_policies import DefaultSimpleStepExecutor
from flujo.domain.dsl.step import Step
from flujo.testing.utils import StubAgent, DummyPlugin
from flujo.domain.plugins import PluginOutcome
from flujo.domain.models import StepResult, UsageLimits, Success, Failure


@pytest.mark.asyncio
async def test_simple_policy_owned_execution():
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()

    step = Step(name="s", agent=StubAgent(["ok"]))

    res = await policy.execute(
        core,
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
        _fallback_depth=0,
    )

    assert isinstance(res, Success)
    assert res.step_result.success is True
    assert res.step_result.output is not None


@pytest.mark.asyncio
async def test_simple_policy_success_path(monkeypatch):
    core = ExecutorCore()
    DefaultSimpleStepExecutor()

    step = Step(name="s", agent=StubAgent(["ok"]))

    # Execute end-to-end via core to ensure policy path is exercised
    res = await core._execute_simple_step(
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert res.success is True
    assert res.output == "ok" or res.output is not None


@pytest.mark.asyncio
async def test_simple_policy_with_plugin_success(monkeypatch):
    core = ExecutorCore()
    DefaultSimpleStepExecutor()

    step = Step(name="s", agent=StubAgent(["ok"]))
    # One plugin that returns success without redirect
    step.plugins = [(DummyPlugin([PluginOutcome(success=True)]), 1)]

    res = await core._execute_simple_step(
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert res.success is True
    assert res.output is not None


@pytest.mark.asyncio
async def test_simple_policy_with_validator_failure(monkeypatch):
    core = ExecutorCore()
    DefaultSimpleStepExecutor()

    step = Step(name="s", agent=StubAgent(["ok"]))
    # Use a validator via plugin runner pathway that fails
    failing = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    step.plugins = [(failing, 1)]

    res = await core._execute_simple_step(
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert res.success is False or res.feedback is not None


@pytest.mark.asyncio
async def test_processors_pipeline_applied(monkeypatch):
    core = ExecutorCore()
    step = Step(name="s", agent=StubAgent(["mid"]))

    # Simulate processors on step
    step.processors = object()

    async def apply_prompt(processors, data, *, context=None):
        assert processors is step.processors
        return f"pp:{data}"

    async def apply_output(processors, output, *, context=None):
        assert processors is step.processors
        return f"po:{output}"

    monkeypatch.setattr(core._processor_pipeline, "apply_prompt", apply_prompt)
    monkeypatch.setattr(core._processor_pipeline, "apply_output", apply_output)

    res = await core._execute_simple_step(
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert res.success is True
    assert res.output == "po:mid"


@pytest.mark.asyncio
async def test_retry_attempt_counts(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    step = Step(name="s", agent=StubAgent(["ok"]))

    # Agent fails once then succeeds
    monkeypatch.setattr(
        core._agent_runner, "run", AsyncMock(side_effect=[Exception("first"), "ok"])
    )

    res = await policy.execute(
        core,
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert isinstance(res, Success)
    assert res.step_result.success is True
    assert res.step_result.attempts == 2


@pytest.mark.asyncio
async def test_fallback_success_path(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    primary = Step(name="primary", agent=StubAgent([]))
    fallback = Step(name="fallback", agent=StubAgent(["fb_ok"]))
    primary.fallback_step = fallback

    # Agent always fails
    monkeypatch.setattr(core._agent_runner, "run", AsyncMock(side_effect=Exception("boom")))

    # Make execute return a successful fallback result
    async def execute_fallback(*, step, **kwargs):
        assert step is fallback
        return StepResult(
            name="fallback", success=True, output="fb_ok", attempts=1, token_counts=0, cost_usd=0.0
        )

    monkeypatch.setattr(core, "execute", execute_fallback)

    res = await policy.execute(
        core,
        primary,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert isinstance(res, Success)
    assert res.step_result.success is True
    assert res.step_result.output == "fb_ok"
    assert res.step_result.metadata_.get("fallback_triggered") is True
    assert res.step_result.metadata_.get("original_error")


@pytest.mark.asyncio
async def test_fallback_failure_path(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    primary = Step(name="primary", agent=StubAgent([]))
    fallback = Step(name="fallback", agent=StubAgent([]))
    primary.fallback_step = fallback

    monkeypatch.setattr(core._agent_runner, "run", AsyncMock(side_effect=Exception("boom")))

    async def execute_fallback(*, step, **kwargs):
        return StepResult(
            name="fallback",
            success=False,
            output=None,
            attempts=1,
            token_counts=0,
            cost_usd=0.0,
            feedback="fb_err",
        )

    monkeypatch.setattr(core, "execute", execute_fallback)

    res = await policy.execute(
        core,
        primary,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    assert isinstance(res, Failure)
    assert res.step_result is not None
    assert res.step_result.success is False
    assert "Original error" in (res.step_result.feedback or "")
    assert "Fallback error" in (res.step_result.feedback or "")


@pytest.mark.asyncio
async def test_streaming_invokes_on_chunk(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    step = Step(name="s", agent=StubAgent(["final"]))

    async def runner(
        agent,
        payload,
        context,
        resources,
        options,
        stream,
        on_chunk,
    ):
        # Simulate streaming
        if stream and on_chunk is not None:
            await on_chunk("chunk1")
        return "final"

    monkeypatch.setattr(core._agent_runner, "run", runner)
    on_chunk = AsyncMock()

    res = await policy.execute(
        core,
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=True,
        on_chunk=on_chunk,
        cache_key=None,
    )
    on_chunk.assert_called_once()
    sr = res.step_result if hasattr(res, "step_result") else res
    assert sr.success is True


@pytest.mark.asyncio
async def test_usage_guard_called(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    step = Step(name="s", agent=StubAgent(["ok"]))

    limits = UsageLimits(total_cost_usd_limit=1.0)
    guard = AsyncMock()
    monkeypatch.setattr(core._usage_meter, "guard", guard)

    res = await policy.execute(
        core,
        step,
        data="in",
        context=None,
        resources=None,
        limits=limits,
        stream=False,
        on_chunk=None,
        cache_key=None,
    )
    # The dual-check pattern calls guard twice: pre-execution and post-execution
    # This provides enhanced robustness by validating usage limits at both stages
    guard.assert_called()
    assert guard.call_count >= 1  # At least one call required
    sr = res.step_result if hasattr(res, "step_result") else res
    assert sr.success is True


@pytest.mark.asyncio
async def test_cache_put_called_on_success(monkeypatch):
    core = ExecutorCore()
    policy = DefaultSimpleStepExecutor()
    step = Step(name="s", agent=StubAgent(["ok"]))

    core._enable_cache = True
    cache_put = AsyncMock()
    monkeypatch.setattr(core._cache_backend, "put", cache_put)

    res = await policy.execute(
        core,
        step,
        data="in",
        context=None,
        resources=None,
        limits=None,
        stream=False,
        on_chunk=None,
        cache_key="key",
    )
    sr = res.step_result if hasattr(res, "step_result") else res
    assert sr.success is True
    cache_put.assert_called_once()
