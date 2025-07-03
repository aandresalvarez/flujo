import time
import pytest

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result


@pytest.mark.asyncio
async def test_caching_pipeline_speed_and_hits() -> None:
    agent = StubAgent(["ok"])
    slow_step = Step.solution(agent)
    cache = InMemoryCache()
    cached = Step.cached(slow_step, cache_backend=cache)

    async def passthrough(x: str) -> str:
        return x

    pipeline = cached >> Step.from_callable(passthrough)
    runner = Flujo(pipeline)

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


@pytest.mark.asyncio
async def test_cache_keys_distinct_for_same_name_steps() -> None:
    a1 = StubAgent(["a", "a"])
    a2 = StubAgent(["b", "b"])
    s1 = Step.cached(Step.solution(a1, name="dup"), cache_backend=InMemoryCache())
    s2 = Step.cached(Step.solution(a2, name="dup"), cache_backend=InMemoryCache())
    runner = Flujo(s1 >> s2)

    await gather_result(runner, "x")
    res = await gather_result(runner, "x")

    h1, h2 = res.step_history
    assert h1.metadata_["cache_hit"] is True
    assert h2.metadata_["cache_hit"] is True
    assert a1.call_count == 1
    assert a2.call_count == 1
