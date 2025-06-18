import pytest
from typing import Any, cast
from flujo import Flujo, Step
from flujo.tracing import ConsoleTracer
from flujo.testing.utils import StubAgent, gather_result
from flujo.domain.agent_protocol import AsyncAgentProtocol


@pytest.mark.asyncio  # type: ignore[misc]
async def test_default_local_tracer_added() -> None:
    step = Step("s", cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    runner = Flujo(step, local_tracer="default")
    assert len(runner.hooks) == 1
    assert callable(runner.hooks[0])


@pytest.mark.asyncio  # type: ignore[misc]
async def test_custom_console_tracer_instance() -> None:
    tracer = ConsoleTracer(level="info")
    step = Step("s", cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    runner = Flujo(step, local_tracer=tracer)
    assert tracer.hook in runner.hooks


@pytest.mark.asyncio  # type: ignore[misc]
async def test_tracer_outputs_info_level(capsys: pytest.CaptureFixture[str]) -> None:
    step = Step("s", cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    runner = Flujo(step, local_tracer="default")
    await gather_result(runner, "in")
    captured = capsys.readouterr().out
    assert "Pipeline Start" in captured
    assert "Step Start" in captured
    assert "Status" in captured


@pytest.mark.asyncio  # type: ignore[misc]
async def test_tracer_outputs_debug_level(capsys: pytest.CaptureFixture[str]) -> None:
    tracer = ConsoleTracer(level="debug")
    step = Step("s", cast(AsyncAgentProtocol[Any, Any], StubAgent(["ok"])))
    runner = Flujo(step, local_tracer=tracer)
    await gather_result(runner, "in")
    captured = capsys.readouterr().out
    assert "Pipeline Start" in captured
    assert "Output" in captured
