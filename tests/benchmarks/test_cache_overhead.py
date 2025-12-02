import time
import pytest

from flujo import Step
from flujo.infra.caching import InMemoryCache
from flujo.testing.utils import StubAgent, gather_result
from tests.conftest import create_test_flujo

pytest.importorskip("pytest_benchmark")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_cache_overhead_vs_plain_step() -> None:
    agent_plain = StubAgent(["ok"] * 5)
    plain = Step.solution(agent_plain)

    agent_cached = StubAgent(["ok"] * 5)
    cached_step = Step.cached(Step.solution(agent_cached), cache_backend=InMemoryCache())

    runner_plain = create_test_flujo(plain, persist_state=False)
    runner_plain.disable_tracing()
    runner_cached = create_test_flujo(cached_step, persist_state=False)
    runner_cached.disable_tracing()

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
