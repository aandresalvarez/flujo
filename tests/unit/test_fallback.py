import pytest
import asyncio


from flujo.domain.dsl import Step, StepConfig
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.application.runner import Flujo, InfiniteFallbackError


@pytest.mark.asyncio
async def test_fallback_assignment() -> None:
    primary = Step.model_validate({"name": "p", "agent": StubAgent(["x"])})
    fb = Step.model_validate({"name": "fb", "agent": StubAgent(["y"])})
    primary.fallback(fb)
    assert primary.fallback_step is fb


@pytest.mark.asyncio
async def test_fallback_not_triggered_on_success() -> None:
    agent = StubAgent(["ok"])
    primary = Step.model_validate({"name": "p", "agent": agent})
    fb = Step.model_validate({"name": "fb", "agent": StubAgent(["fb"])})
    primary.fallback(fb)
    runner = Flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]
    assert sr.output == "ok"
    assert agent.call_count == 1
    assert getattr(fb.agent, "call_count", 0) == 0
    assert sr.metadata_ is None


@pytest.mark.asyncio
async def test_fallback_triggered_on_failure() -> None:
    primary_agent = StubAgent(["bad"])
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": primary_agent,
            "config": StepConfig(max_retries=1),
            "plugins": [(plugin, 0)],
        }
    )
    fb_agent = StubAgent(["recover"])
    fb = Step.model_validate({"name": "fb", "agent": fb_agent})
    primary.fallback(fb)
    runner = Flujo(primary)
    res = await gather_result(runner, "data")
    sr = res.step_history[0]
    assert sr.success is True
    assert sr.output == "recover"
    assert sr.metadata_ and sr.metadata_["fallback_triggered"] is True
    assert primary_agent.call_count == 1
    assert fb_agent.call_count == 1


@pytest.mark.asyncio
async def test_fallback_failure_propagates() -> None:
    primary_agent = StubAgent(["bad"])
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback="p fail")])
    primary = Step.model_validate(
        {"name": "p", "agent": primary_agent, "plugins": [(plugin_primary, 0)]}
    )
    fb_agent = StubAgent(["still bad"])
    plugin_fb = DummyPlugin([PluginOutcome(success=False, feedback="fb fail")])
    fb = Step.model_validate({"name": "fb", "agent": fb_agent, "plugins": [(plugin_fb, 0)]})
    primary.fallback(fb)
    runner = Flujo(primary)
    res = await gather_result(runner, "data")
    sr = res.step_history[0]
    assert sr.success is False
    assert "p fail" in sr.feedback
    assert "fb fail" in sr.feedback
    assert fb_agent.call_count == 1


class WrappedResult:
    def __init__(self, output: str, token_counts: int = 2, cost_usd: float = 0.1) -> None:
        self.output = output
        self.token_counts = token_counts
        self.cost_usd = cost_usd


class SlowAgent:
    async def run(self, data: str) -> WrappedResult:
        await asyncio.sleep(0.05)
        return WrappedResult("slow")


@pytest.mark.asyncio
async def test_fallback_latency_accumulated() -> None:
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    failing = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": SlowAgent()})
    failing.fallback(fb)
    runner = Flujo(failing)
    res = await gather_result(runner, "x")
    sr = res.step_history[0]
    assert sr.success is True
    assert sr.latency_s >= 0.05


class CostlyOutput:
    def __init__(self, output: str) -> None:
        self.output = output
        self.token_counts = 5
        self.cost_usd = 0.2


@pytest.mark.asyncio
async def test_failed_fallback_accumulates_metrics() -> None:
    plugin_primary = DummyPlugin([PluginOutcome(success=False, feedback="bad")])
    primary = Step.model_validate(
        {
            "name": "p",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin_primary, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_plugin = DummyPlugin([PluginOutcome(success=False, feedback="worse")])
    fb_agent = StubAgent([CostlyOutput("oops")])
    fb = Step.model_validate({"name": "fb", "agent": fb_agent, "plugins": [(fb_plugin, 0)]})
    primary.fallback(fb)
    runner = Flujo(primary)
    res = await gather_result(runner, "in")
    sr = res.step_history[0]
    assert sr.success is False
    assert sr.cost_usd == 0.2
    assert sr.token_counts == 6


@pytest.mark.asyncio
async def test_infinite_fallback_loop_detected() -> None:
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    a = Step.model_validate(
        {
            "name": "a",
            "agent": StubAgent(["bad"] * 100),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    b = Step.model_validate(
        {
            "name": "b",
            "agent": StubAgent(["bad"] * 100),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    a.fallback(b)
    b.fallback(a)
    runner = Flujo(a)
    with pytest.raises(InfiniteFallbackError):
        await gather_result(runner, "data")
