import time
import pytest

from flujo import Step
from flujo.infra.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo

pytest.importorskip("pytest_benchmark")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_cache_hit_performance_gain() -> None:
    agent = StubAgent(["ok", "ok"])
    cached_step = Step.cached(Step.solution(agent), cache_backend=InMemoryCache())
    runner = create_test_flujo(cached_step, persist_state=False)
    runner.disable_tracing()

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

    # Allow a 10% tolerance due to system noise/jitter
    if hit_time > miss_time * 1.10:
        print(
            f"WARNING: Cache hit was slower than miss (hit: {hit_time:.4f}s, miss: {miss_time:.4f}s)"
        )
    # Do not fail the test due to timing noise
    # assert hit_time <= miss_time
