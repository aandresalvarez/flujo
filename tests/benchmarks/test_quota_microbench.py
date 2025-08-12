import time
import threading
from typing import List

from flujo.domain.models import Quota, UsageEstimate


def _reserve_loop(q: Quota, estimates: List[UsageEstimate], iterations: int) -> None:
    for _ in range(iterations):
        for est in estimates:
            q.reserve(est)
            q.reclaim(est, est)


def test_quota_reservation_overhead_microbenchmark() -> None:
    # Simple smoke micro-benchmark; not asserting timing to avoid flakiness in CI
    estimates = [UsageEstimate(cost_usd=0.001, tokens=1) for _ in range(100)]

    # Single-thread
    q1 = Quota(remaining_cost_usd=1e6, remaining_tokens=1_000_000)
    t0 = time.perf_counter()
    _reserve_loop(q1, estimates, 100)
    single_duration = time.perf_counter() - t0

    # Multi-thread (contention)
    q2 = Quota(remaining_cost_usd=1e6, remaining_tokens=1_000_000)
    threads = [threading.Thread(target=_reserve_loop, args=(q2, estimates, 100)) for _ in range(8)]
    t1 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    multi_duration = time.perf_counter() - t1

    # Ensure code executes and durations are finite
    assert single_duration >= 0.0
    assert multi_duration >= 0.0
