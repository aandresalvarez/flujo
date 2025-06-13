import asyncio
from unittest.mock import Mock
import pytest

from pydantic_ai_orchestrator.domain import Step
from pydantic_ai_orchestrator.application.pipeline_runner import (
    PipelineRunner,
    PipelineResult,
)
from pydantic_ai_orchestrator.testing.utils import StubAgent, DummyPlugin
from pydantic_ai_orchestrator.domain.plugins import PluginOutcome


async def test_runner_respects_max_retries():
    agent = StubAgent(["a", "b", "c"])
    plugin = DummyPlugin([
        PluginOutcome(success=False),
        PluginOutcome(success=False),
        PluginOutcome(success=True),
    ])
    step = Step("test", agent, max_retries=3, plugins=[plugin])
    pipeline = step
    runner = PipelineRunner(pipeline)
    result = await runner.run_async("in")
    assert agent.call_count == 3
    assert isinstance(result, PipelineResult)
    assert result.step_history[0].attempts == 3


async def test_feedback_enriches_prompt():
    sol_agent = StubAgent(["sol1", "sol2"])
    plugin = DummyPlugin([
        PluginOutcome(success=False, feedback="SQL Error: XYZ"),
        PluginOutcome(success=True),
    ])
    step = Step.solution(sol_agent, max_retries=2, plugins=[plugin])
    runner = PipelineRunner(step)
    await runner.run_async("SELECT *")
    assert sol_agent.call_count == 2
    assert "SQL Error: XYZ" in sol_agent.inputs[1]


async def test_conditional_redirection():
    primary = StubAgent(["first"])
    fixit = StubAgent(["fixed"])
    plugin = DummyPlugin([
        PluginOutcome(success=False, redirect_to=fixit),
        PluginOutcome(success=True),
    ])
    step = Step("s", primary, max_retries=2, plugins=[plugin])
    pipeline = step
    runner = PipelineRunner(pipeline)
    await runner.run_async("prompt")
    assert primary.call_count == 1
    assert fixit.call_count == 1


async def test_on_failure_called():
    agent = StubAgent(["out"])
    plugin = DummyPlugin([PluginOutcome(success=False)])
    handler = Mock()
    step = Step("s", agent, max_retries=1, plugins=[plugin])
    step.on_failure(handler)
    runner = PipelineRunner(step)
    await runner.run_async("prompt")
    handler.assert_called_once()


async def test_timeout_and_redirect_loop_detection():
    async def slow_validate(data):
        await asyncio.sleep(0.05)
        return PluginOutcome(success=True)

    class SlowPlugin:
        async def validate(self, data):
            return await slow_validate(data)

    plugin = SlowPlugin()
    agent = StubAgent(["ok"])
    step = Step("s", agent, plugins=[plugin], max_retries=1, timeout_s=0.01)
    runner = PipelineRunner(step)
    try:
        await runner.run_async("prompt")
    except TimeoutError:
        pass

    # Redirect loop
    a1 = StubAgent(["a1"])
    a2 = StubAgent(["a2"])
    plugin_loop = DummyPlugin([
        PluginOutcome(success=False, redirect_to=a2),
        PluginOutcome(success=False, redirect_to=a1),
    ])
    step2 = Step("loop", a1, max_retries=3, plugins=[plugin_loop])
    runner2 = PipelineRunner(step2)
    with pytest.raises(Exception):
        await runner2.run_async("p")
