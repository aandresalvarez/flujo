import pytest
import time

from flujo.application.conversation.history_manager import HistoryManager, HistoryStrategyConfig
from flujo.domain.models import ConversationTurn, ConversationRole


@pytest.mark.benchmark
@pytest.mark.slow
def test_history_manager_overhead_benchmark(monkeypatch):
    # Prepare a moderate history
    turns = [ConversationTurn(role=ConversationRole.user, content=f"u{i}") for i in range(50)] + [
        ConversationTurn(role=ConversationRole.assistant, content=f"a{i}") for i in range(50)
    ]
    hm = HistoryManager(HistoryStrategyConfig(strategy="truncate_tokens", max_tokens=2048))
    # Measure 100 iterations of bound_history
    iters = 100
    start = time.perf_counter()
    for _ in range(iters):
        _ = hm.bound_history(turns, model_id=None)
    elapsed = time.perf_counter() - start
    per_iter = elapsed / iters
    # Not asserting strict thresholds due to CI variability; ensure it runs quickly
    # A sanity upper bound (e.g., < 1.0s) to catch extreme regressions
    assert elapsed < 1.0
    # Strict performance gate: enable with FLUJO_STRICT_PERF=1
    import os

    if str(os.environ.get("FLUJO_STRICT_PERF", "")).strip() in {"1", "true", "on", "yes"}:
        assert per_iter < 0.005, f"Per-iteration overhead too high: {per_iter * 1000:.2f} ms"
