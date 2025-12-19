import pytest

from flujo.application.runner import Flujo
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step


def make_counting(counter: dict):
    async def _counting(x: object) -> object:
        counter["n"] = counter.get("n", 0) + 1
        return x

    return _counting


@pytest.mark.asyncio
async def test_cache_yaml_hits_on_second_run():
    yaml_text = """
version: "0.1"
name: "cache-yaml"
steps:
  - kind: cache
    name: CachedStringify
    wrapped_step:
      kind: step
      name: ToString
      agent: { id: "flujo.builtins.stringify" }
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
    r = Flujo(pipeline=pipeline)

    # First run (no cache)
    first = None
    async for item in r.run_async("foo"):
        first = item
    # Second run (should be served from cache)
    second = None
    async for item in r.run_async("foo"):
        second = item

    # Check for cache hit on second run
    assert second.step_history, "expected step history for cache check"
    last = second.step_history[-1]
    assert last.metadata_.get("cache_hit") is True
    # Output should be identical
    assert last.output == first.step_history[-1].output


@pytest.mark.asyncio
async def test_cache_prevents_double_invocation_of_agent():
    # Build cached step programmatically to count invocations
    counter = {"n": 0}
    step = Step.from_callable(make_counting(counter), name="Counting")
    from flujo.domain.dsl.cache_step import CacheStep

    cached = CacheStep.cached(step)
    pipeline = Pipeline(steps=[cached])
    r = Flujo(pipeline=pipeline)

    # First run increments counter
    async for item in r.run_async("x"):
        pass
    assert counter["n"] == 1

    # Second run should hit cache and not increment counter
    async for item in r.run_async("x"):
        pass
    assert counter["n"] == 1, "agent should not be reinvoked on cache hit"
