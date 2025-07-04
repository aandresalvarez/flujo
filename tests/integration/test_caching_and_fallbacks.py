import time
import pytest

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result, DummyPlugin
from flujo.domain import Pipeline
from flujo.domain.pipeline_dsl import StepConfig
from flujo.domain.plugins import PluginOutcome
from typing import Any


@pytest.mark.asyncio  # type: ignore[misc]
async def test_caching_pipeline_speed_and_hits() -> None:
    agent: StubAgent = StubAgent(["ok"])
    slow_step: Step[Any, Any] = Step.solution(agent)
    cache: InMemoryCache = InMemoryCache()
    cached: Step[Any, Any] = Step.cached(slow_step, cache_backend=cache)

    async def passthrough(x: str) -> str:
        return x

    pipeline: Pipeline[Any, Any] = cached >> Step.from_callable(passthrough)
    runner: Flujo[Any, Any, Any] = Flujo(pipeline)

    start = time.monotonic()
    first = await gather_result(runner, "a")
    first_meta = first.step_history[0].metadata_
    first_time = time.monotonic() - start

    start = time.monotonic()
    result2 = await gather_result(runner, "a")
    second_time = time.monotonic() - start

    assert second_time <= first_time
    assert result2.step_history[0].metadata_["cache_hit"] is True

    start = time.monotonic()
    result3 = await gather_result(runner, "b")
    third_time = time.monotonic() - start

    assert first_meta is None or "cache_hit" not in first_meta
    assert (
        result3.step_history[0].metadata_ is None
        or "cache_hit" not in result3.step_history[0].metadata_
    )
    assert agent.call_count == 2
    assert third_time >= 0


@pytest.mark.asyncio  # type: ignore[misc]
async def test_cache_keys_distinct_for_same_name_steps() -> None:
    a1: StubAgent = StubAgent(["a", "a"])
    a2: StubAgent = StubAgent(["b", "b"])
    s1: Step[Any, Any] = Step.cached(Step.solution(a1, name="dup"), cache_backend=InMemoryCache())
    s2: Step[Any, Any] = Step.cached(Step.solution(a2, name="dup"), cache_backend=InMemoryCache())
    runner: Flujo[Any, Any, Any] = Flujo(s1 >> s2)

    await gather_result(runner, "x")
    res = await gather_result(runner, "x")

    h1, h2 = res.step_history
    assert h1.metadata_["cache_hit"] is True
    assert h2.metadata_["cache_hit"] is True
    assert a1.call_count == 1
    assert a2.call_count == 1


@pytest.mark.asyncio  # type: ignore[misc]
async def test_pipeline_step_fallback() -> None:
    s1: Step[Any, Any] = Step.model_validate({"name": "s1", "agent": StubAgent(["a"])})
    plugin: DummyPlugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    failing: Step[Any, Any] = Step.model_validate(
        {
            "name": "s2",
            "agent": StubAgent(["bad"]),
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb: Step[Any, Any] = Step.model_validate({"name": "fb", "agent": StubAgent(["good"])})
    failing.fallback(fb)
    s3: Step[Any, Any] = Step.model_validate({"name": "s3", "agent": StubAgent(["end"])})
    pipeline: Pipeline[Any, Any] = s1 >> failing >> s3
    result = await gather_result(Flujo(pipeline), "in")
    assert result.step_history[1].output == "good"
    assert result.step_history[1].metadata_["fallback_triggered"] is True
    assert result.step_history[2].output == "end"


@pytest.mark.asyncio  # type: ignore[misc]
async def test_loop_step_fallback_continues() -> None:
    body_agent: StubAgent = StubAgent(["bad", "done"])
    plugin: DummyPlugin = DummyPlugin(
        [PluginOutcome(success=False, feedback="err"), PluginOutcome(success=True)]
    )
    body: Step[Any, Any] = Step.model_validate(
        {
            "name": "body",
            "agent": body_agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_agent: StubAgent = StubAgent(["recover"])
    fb: Step[Any, Any] = Step.model_validate({"name": "fb", "agent": fb_agent})
    body.fallback(fb)
    loop_step: Step[Any, Any] = Step.loop_until(
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


@pytest.mark.asyncio  # type: ignore[misc]
async def test_conditional_branch_with_fallback() -> None:
    branch_agent: StubAgent = StubAgent(["bad"])
    plugin: DummyPlugin = DummyPlugin([PluginOutcome(success=False, feedback="err")])
    branch_step: Step[Any, Any] = Step.model_validate(
        {
            "name": "branch",
            "agent": branch_agent,
            "plugins": [(plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    fb_agent: StubAgent = StubAgent(["fix"])
    branch_step.fallback(Step.model_validate({"name": "branch_fb", "agent": fb_agent}))

    cond: Step[Any, Any] = Step.branch_on(
        name="cond",
        condition_callable=lambda *_: "a",
        branches={"a": Pipeline.from_step(branch_step)},
    )
    final: Step[Any, Any] = Step.model_validate({"name": "final", "agent": StubAgent(["end"])})
    pipeline: Pipeline[Any, Any] = cond >> final
    result = await gather_result(Flujo(pipeline), "x")
    assert fb_agent.call_count == 1
    assert result.step_history[-1].output == "end"
