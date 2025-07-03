from flujo.domain import Step, Pipeline
from flujo.domain.pipeline_dsl import StepConfig
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.application.flujo_engine import Flujo
import pytest


@pytest.mark.asyncio
async def test_pipeline_step_fallback() -> None:
    s1 = Step.model_validate({"name": "s1", "agent": StubAgent(["a"])})
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    failing = Step.model_validate(
        {
            "name": "s2",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb = Step.model_validate({"name": "fb", "agent": StubAgent(["good"])})
    failing.fallback(fb)
    s3 = Step.model_validate({"name": "s3", "agent": StubAgent(["end"])})
    pipeline = s1 >> failing >> s3
    result = await gather_result(Flujo(pipeline), "in")
    assert result.step_history[1].output == "good"
    assert result.step_history[1].metadata_["fallback_triggered"] is True
    assert result.step_history[2].output == "end"


@pytest.mark.asyncio
async def test_loop_step_fallback_continues() -> None:
    body_agent = StubAgent(["bad", "done"])
    plugin = DummyPlugin(
        [PluginOutcome(success=False, feedback="err"), PluginOutcome(success=True)]
    )
    body = Step.model_validate(
        {
            "name": "body",
            "agent": body_agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_agent = StubAgent(["recover"])
    fb = Step.model_validate({"name": "fb", "agent": fb_agent})
    body.fallback(fb)
    loop_step = Step.loop_until(
        name="loop",
        loop_body_pipeline=Pipeline.from_step(body),
        exit_condition_callable=lambda out, _ctx: out == "done",
        max_loops=2,
    )
    result = await gather_result(Flujo(loop_step), "start")
    sr = result.step_history[0]
    assert sr.success is True
    assert body_agent.call_count == 2
    assert fb_agent.call_count == 1
    assert sr.output == "done"


@pytest.mark.asyncio
async def test_conditional_branch_with_fallback() -> None:
    branch_agent = StubAgent(["bad"])
    plugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    branch_step = Step.model_validate(
        {
            "name": "branch",
            "agent": branch_agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_agent = StubAgent(["fix"])
    branch_step.fallback(Step.model_validate({"name": "branch_fb", "agent": fb_agent}))

    cond = Step.branch_on(
        name="cond",
        condition_callable=lambda *_: "a",
        branches={"a": Pipeline.from_step(branch_step)},
    )
    final = Step.model_validate({"name": "final", "agent": StubAgent(["end"])})
    pipeline = cond >> final
    result = await gather_result(Flujo(pipeline), "x")
    assert fb_agent.call_count == 1
    assert result.step_history[-1].output == "end"
