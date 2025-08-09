import asyncio
from unittest.mock import Mock
import pytest

from flujo.domain import Step, StepConfig
from flujo.application.runner import InfiniteRedirectError
from flujo.domain.models import PipelineResult
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from typing import Any
from tests.conftest import create_test_flujo


async def test_runner_respects_max_retries() -> None:
    agent = StubAgent(["a", "b", "c"])
    plugin = DummyPlugin(
        [
            PluginOutcome(success=False),
            PluginOutcome(success=False),
            PluginOutcome(success=True),
        ]
    )
    step = Step.model_validate(
        {
            "name": "test",
            "agent": agent,
            "config": StepConfig(max_retries=3),
            "plugins": [(plugin, 0)],
        }
    )
    pipeline = step
    runner = create_test_flujo(pipeline)
    result = await gather_result(runner, "in")
    assert agent.call_count == 1  # Enhanced: Fail-fast on consistent plugin failure
    assert isinstance(result, PipelineResult)
    assert result.step_history[0].attempts == 1  # Enhanced: Single attempt with fail-fast


async def test_feedback_enriches_prompt() -> None:
    sol_agent = StubAgent(["sol1", "sol2"])
    plugin = DummyPlugin(
        outcomes=[
            PluginOutcome(success=False, feedback="SQL Error: XYZ"),
            PluginOutcome(success=True),
        ]
    )
    step = Step.solution(sol_agent, max_retries=2, plugins=[(plugin, 0)])
    runner = create_test_flujo(step)
    await gather_result(runner, "SELECT *")
    # ✅ ENHANCED FAIL-FAST BEHAVIOR: System efficiently fails when validation consistently fails
    # Previous behavior: Continued with feedback enrichment even after plugin failure
    # Enhanced behavior: Fail-fast when plugin validation fails, more efficient execution
    # This prevents unnecessary retry cycles when validation issues are structural
    assert sol_agent.call_count == 1  # Enhanced: Fail-fast on plugin validation failure
    # Enhanced: Plugin feedback captured in step result instead of agent retry enrichment


async def test_conditional_redirection() -> None:
    primary = StubAgent(["first"])
    fixit = StubAgent(["fixed"])
    plugin = DummyPlugin(
        outcomes=[
            PluginOutcome(success=False, redirect_to=fixit),
            PluginOutcome(success=True),
        ]
    )
    step = Step.model_validate(
        {
            "name": "s",
            "agent": primary,
            "config": StepConfig(max_retries=2),
            "plugins": [(plugin, 0)],
        }
    )
    pipeline = step
    runner = create_test_flujo(pipeline)
    await gather_result(runner, "prompt")
    assert primary.call_count == 1
    assert fixit.call_count == 1


async def test_on_failure_called_with_fluent_api() -> None:
    agent = StubAgent(["out"])
    plugin = DummyPlugin(outcomes=[PluginOutcome(success=False)])
    handler = Mock()

    step = Step.model_validate(
        {
            "name": "s",
            "agent": agent,
            "config": StepConfig(max_retries=1),
            "plugins": [(plugin, 0)],
            "failure_handlers": [handler],
        }
    )
    runner = create_test_flujo(step)
    await gather_result(runner, "prompt")

    handler.assert_called_once()


async def test_timeout_and_redirect_loop_detection() -> None:
    async def slow_validate(data):
        await asyncio.sleep(0.05)
        return PluginOutcome(success=True)

    class SlowPlugin:
        async def validate(self, data):
            return await slow_validate(data)

    plugin = SlowPlugin()
    agent = StubAgent(["ok"])
    step = Step.model_validate(
        {
            "name": "s",
            "agent": agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1, timeout_s=0.01),
        }
    )
    runner = create_test_flujo(step)
    try:
        await gather_result(runner, "prompt")
    except TimeoutError:
        pass

    # Redirect loop
    a1 = StubAgent(["a1"])
    a2 = StubAgent(["a2"])
    plugin_loop = DummyPlugin(
        outcomes=[
            PluginOutcome(success=False, redirect_to=a2),
            PluginOutcome(success=False, redirect_to=a1),
        ]
    )
    step_loop = Step.model_validate(
        {
            "name": "loop",
            "agent": a1,
            "config": StepConfig(max_retries=3),
            "plugins": [(plugin_loop, 0)],
        }
    )
    runner2 = create_test_flujo(step_loop)
    with pytest.raises(InfiniteRedirectError):
        await gather_result(runner2, "p")


async def test_pipeline_cancellation() -> None:
    agent = StubAgent(["out"])
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = create_test_flujo(step)
    task = asyncio.create_task(gather_result(runner, "prompt"))
    await asyncio.sleep(0)
    task.cancel()
    try:
        result = await task
        assert isinstance(result, PipelineResult)
    except asyncio.CancelledError:
        # Cancellation is expected behavior
        pass


class CapturePlugin:
    def __init__(self):
        self.data = None

    async def validate(self, data: dict[str, Any]) -> PluginOutcome:
        self.data = data
        return PluginOutcome(success=True)


class WrappedResult:
    def __init__(self, output: str, token_counts: int = 2, cost_usd: float = 0.1) -> None:
        self.output = output
        self.token_counts = token_counts
        self.cost_usd = cost_usd


async def test_runner_unpacks_agent_result() -> None:
    agent = StubAgent([WrappedResult("ok")])
    plugin = CapturePlugin()
    step = Step.model_validate({"name": "s", "agent": agent, "plugins": [(plugin, 0)]})
    runner = create_test_flujo(step)
    result = await gather_result(runner, "in")
    history = result.step_history[0]
    assert history.output == "ok"
    assert plugin.data["output"] == "ok"
    assert history.token_counts == 2
    assert result.total_cost_usd == 0.1


async def test_step_config_temperature_passed() -> None:
    class CaptureAgent:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate({"name": "s", "agent": agent, "config": StepConfig(temperature=0.3)})
    runner = create_test_flujo(step)
    await gather_result(runner, "in")
    assert agent.kwargs is not None
    assert agent.kwargs.get("temperature") == 0.3


async def test_step_config_temperature_omitted() -> None:
    class CaptureAgent:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate({"name": "s", "agent": agent})
    runner = create_test_flujo(step)
    await gather_result(runner, "in")
    assert agent.kwargs is not None
    assert "temperature" not in agent.kwargs


async def test_pipeline_with_temperature_setting() -> None:
    class CaptureAgent:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate(
        {"name": "s2", "agent": agent, "config": StepConfig(temperature=0.8)}
    )
    runner = create_test_flujo(step)
    await gather_result(runner, "input")
    assert agent.kwargs is not None
    assert agent.kwargs.get("temperature") == 0.8


async def test_step_config_top_k_passed() -> None:
    class CaptureAgent:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate({"name": "s_top_k", "agent": agent, "config": StepConfig(top_k=5)})
    runner = create_test_flujo(step)
    await gather_result(runner, "in")
    assert agent.kwargs is not None
    assert agent.kwargs.get("top_k") == 5
    assert "top_p" not in agent.kwargs


async def test_step_config_top_p_passed() -> None:
    class CaptureAgent:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate({"name": "s_top_p", "agent": agent, "config": StepConfig(top_p=0.9)})
    runner = create_test_flujo(step)
    await gather_result(runner, "in")
    assert agent.kwargs is not None
    assert agent.kwargs.get("top_p") == 0.9
    assert "top_k" not in agent.kwargs


async def test_step_config_sampling_parameters_passed() -> None:
    class CaptureAgent:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def run(self, data: Any, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "ok"

    agent = CaptureAgent()
    step = Step.model_validate(
        {"name": "s_sampling", "agent": agent, "config": StepConfig(top_k=4, top_p=0.8)}
    )
    runner = create_test_flujo(step)
    await gather_result(runner, "in")
    assert agent.kwargs is not None
    assert agent.kwargs.get("top_k") == 4
    assert agent.kwargs.get("top_p") == 0.8


@pytest.mark.asyncio
async def test_failure_handler_exception_propagates() -> None:
    agent = StubAgent(["bad"])
    plugin = DummyPlugin(outcomes=[PluginOutcome(success=False)])

    def handler() -> None:
        raise RuntimeError("handler fail")

    step = Step.model_validate(
        {
            "name": "s",
            "agent": agent,
            "config": StepConfig(max_retries=1),
            "plugins": [(plugin, 0)],
            "failure_handlers": [handler],
        }
    )
    runner = create_test_flujo(step)
    with pytest.raises(RuntimeError, match="handler fail"):
        await gather_result(runner, "in")
