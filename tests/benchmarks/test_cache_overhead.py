import time
import pytest

from flujo import Step, Flujo
from flujo.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result

pytest.importorskip("pytest_benchmark")


@pytest.mark.asyncio
async def test_cache_overhead_vs_plain_step() -> None:
    agent_plain = StubAgent(["ok"] * 5)
    plain = Step.solution(agent_plain)

    agent_cached = StubAgent(["ok"] * 5)
    cached_step = Step.cached(Step.solution(agent_cached), cache_backend=InMemoryCache())

    runner_plain = Flujo(plain)
    runner_cached = Flujo(cached_step)

    start = time.monotonic()
    await gather_result(runner_plain, "in")
    plain_time = time.monotonic() - start

    start = time.monotonic()
    await gather_result(runner_cached, "in")
    cached_time = time.monotonic() - start

    print("\nCache miss overhead results:")
    print(f"Plain step time: {plain_time:.4f}s")
    print(f"Cached step miss time: {cached_time:.4f}s")

    assert plain_time >= 0
    assert cached_time >= 0
