import pytest

from flujo import Step, Flujo
from flujo.infra.caching import InMemoryCache
from flujo.testing.utils import StubAgent, DummyPlugin, gather_result
from flujo.domain.plugins import PluginOutcome
from flujo.domain.dsl import StepConfig
from typing import Any


@pytest.mark.asyncio
async def test_cached_fallback_result_is_reused() -> None:
    # Primary step always fails via plugin
    primary_agent: StubAgent = StubAgent(["bad"])
    failing_plugin: DummyPlugin = DummyPlugin(
        outcomes=[PluginOutcome(success=False, feedback="err")]
    )
    primary_step: Step[Any, Any] = Step.model_validate(
        {
            "name": "primary",
            "agent": primary_agent,
            "plugins": [(failing_plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )

    # Fallback step succeeds
    fallback_agent: StubAgent = StubAgent(["fallback_success_output"])
    fallback_step: Step[Any, Any] = Step.model_validate({"name": "fb", "agent": fallback_agent})
    primary_step.fallback(fallback_step)

    cached_step: Step[Any, Any] = Step.cached(primary_step, cache_backend=InMemoryCache())
    runner: Flujo[Any, Any, Any] = Flujo(cached_step)

    # First run triggers fallback and caches its result
    result1 = await gather_result(runner, "initial_input")
    sr1 = result1.step_history[0]
    assert sr1.output == "fallback_success_output"
    assert sr1.metadata_["fallback_triggered"] is True
    assert "cache_hit" not in (sr1.metadata_ or {})
    assert primary_agent.call_count == 2  # Enhanced: Plugin failure triggers retry before fallback
    assert fallback_agent.call_count == 1

    # Second run should hit the cache
    result2 = await gather_result(runner, "initial_input")
    sr2 = result2.step_history[0]
    assert sr2.output == "fallback_success_output"
    assert sr2.metadata_["cache_hit"] is True
    assert primary_agent.call_count == 2  # Enhanced: Count preserved from first run
    assert fallback_agent.call_count == 1


@pytest.mark.asyncio
async def test_no_cache_when_fallback_fails() -> None:
    primary_agent: StubAgent = StubAgent(["bad", "bad"])
    failing_plugin: DummyPlugin = DummyPlugin(
        outcomes=[
            PluginOutcome(success=False, feedback="err"),
            PluginOutcome(success=False, feedback="err"),
        ]
    )
    primary_step: Step[Any, Any] = Step.model_validate(
        {
            "name": "primary",
            "agent": primary_agent,
            "plugins": [(failing_plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )

    fallback_agent: StubAgent = StubAgent(["fb_bad", "fb_bad"])
    fallback_plugin: DummyPlugin = DummyPlugin(
        outcomes=[PluginOutcome(success=False, feedback="err")]
    )
    fallback_step: Step[Any, Any] = Step.model_validate(
        {
            "name": "fb",
            "agent": fallback_agent,
            "plugins": [(fallback_plugin, 0)],
            "config": StepConfig(max_retries=1),
        }
    )
    primary_step.fallback(fallback_step)

    cached_step: Step[Any, Any] = Step.cached(primary_step, cache_backend=InMemoryCache())
    runner: Flujo[Any, Any, Any] = Flujo(cached_step)

    # First run - both primary and fallback fail
    result1 = await gather_result(runner, "x")
    sr1 = result1.step_history[0]
    assert sr1.success is False
    assert "cache_hit" not in (sr1.metadata_ or {})

    # Second run - should not be cache hit and agents run again
    result2 = await gather_result(runner, "x")
    sr2 = result2.step_history[0]
    assert sr2.success is False
    assert "cache_hit" not in (sr2.metadata_ or {})
    assert primary_agent.call_count == 4  # Enhanced: 2 retries per run due to plugin failures
    assert fallback_agent.call_count == 2
