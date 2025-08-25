import pytest
from typing import Any, cast
from flujo import Step
from flujo.infra.console_tracer import ConsoleTracer
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.agent_protocol import AsyncAgentProtocol
from tests.conftest import create_test_flujo


@pytest.mark.asyncio
async def test_default_local_tracer_added() -> None:
    step = Step.model_validate(
        {"name": "s", "agent": cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))}
    )
    runner = create_test_flujo(step, local_tracer="default")
    # In test_mode, tracing is disabled by default, so only ConsoleTracer is present.
    # When tracing is enabled, TraceManager is also present. Assert flexibly.
    assert len(runner.hooks) >= 1
    assert all(callable(h) for h in runner.hooks)
    # Ensure a ConsoleTracer hook is registered
    assert any(
        getattr(h, "__self__", None).__class__.__name__ == "ConsoleTracer" for h in runner.hooks
    )


@pytest.mark.asyncio
async def test_custom_console_tracer_instance() -> None:
    tracer = ConsoleTracer(level="info")
    step = Step.model_validate(
        {"name": "s", "agent": cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))}
    )
    runner = create_test_flujo(step, local_tracer=tracer)
    assert tracer.hook in runner.hooks


@pytest.mark.asyncio
async def test_tracer_outputs_info_level(capsys: pytest.CaptureFixture[str]) -> None:
    step = Step.model_validate(
        {"name": "s", "agent": cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))}
    )
    runner = create_test_flujo(step, local_tracer="default")
    await gather_result(runner, "in")
    captured = capsys.readouterr().out
    assert "Pipeline Start" in captured
    assert "Step Start" in captured
    assert "Status" in captured


@pytest.mark.asyncio
async def test_tracer_outputs_debug_level(capsys: pytest.CaptureFixture[str]) -> None:
    tracer = ConsoleTracer(level="debug")
    step = Step.model_validate(
        {"name": "s", "agent": cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"]))}
    )
    runner = create_test_flujo(step, local_tracer=tracer)
    await gather_result(runner, "in")
    captured = capsys.readouterr().out
    assert "Pipeline Start" in captured
    assert "Output" in captured
