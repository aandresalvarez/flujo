import time
import pytest

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result

pytest.importorskip("pytest_benchmark")


@pytest.mark.asyncio
async def test_cache_hit_performance_gain() -> None:
    agent = StubAgent(["ok"])
    cached_step = Step.cached(Step.solution(agent), cache_backend=InMemoryCache())
    runner = Flujo(cached_step)

    start = time.monotonic()
    await gather_result(runner, "x")
    miss_time = time.monotonic() - start

    start = time.monotonic()
    result = await gather_result(runner, "x")
    hit_time = time.monotonic() - start

    assert result.step_history[0].metadata_["cache_hit"] is True
    print("\nCache hit performance results:")
    print(f"Miss time: {miss_time:.4f}s")
    print(f"Hit time: {hit_time:.4f}s")

    assert hit_time <= miss_time
