import pytest


from flujo.domain.pipeline_dsl import Step, StepConfig
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.application.flujo_engine import Flujo


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
    assert fb.agent.call_count == getattr(fb.agent, "call_count", 0)
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
