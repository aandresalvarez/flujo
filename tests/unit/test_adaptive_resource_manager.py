import asyncio
import math
import time
from collections import deque

import pytest

try:
    # Prefer pytest-asyncio if available
    import pytest_asyncio  # noqa: F401
    ASYNC_MARK = pytest.mark.asyncio
except Exception:
    # Fallback to pytest's asyncio marker (some repos alias it)
    ASYNC_MARK = pytest.mark.asyncio

# Attempt to import the module under test. Adjust the import path if module name differs.
# We try common locations; tests should be run in repo context where package is importable.
# If import path is different in this repo, update below accordingly.
try:
    # If the classes live in a package (e.g., "src" with PEP 420), ensure PYTHONPATH includes it via test runner config.
    from adaptive_resource_manager import (  # type: ignore
        ResourceType,
        PressureLevel,
        AdaptationStrategy,
        ResourceMetrics,
        ResourceLimit,
        AdaptationEvent,
        SystemMonitor,
        AdaptiveResourceManager,
        get_global_adaptive_resource_manager,
        set_resource_limit,
    )
except Exception:
    # Fallback: some repos nest under package like 'app.optimization' or similar
    # We deliberately re-raise with clear message to help adjust path quickly.
    raise ImportError(
        "Unable to import adaptive_resource_manager module. "
        "Please adjust the import path in tests/unit/test_adaptive_resource_manager.py to match project package structure."
    )

class DummyTelemetry:
    def __init__(self):
        self.metrics = []
        self.counters = []

    def record_metric(self, name, value, tags=None):
        self.metrics.append((name, value, tags or {}))

    def increment_counter(self, name, value=1, tags=None):
        self.counters.append((name, value, tags or {}))

class DummyMemoryOptimizer:
    def __init__(self):
        self.tracked = []

    def track_object(self, obj, name):
        self.tracked.append((name, obj))

@pytest.fixture
def fake_time(monkeypatch):
    # Provide deterministic time for AdaptationEvent timestamps and pressure timing
    t = {"now": 1_700_000_000.0}
    def _time():
        return t["now"]
    monkeypatch.setattr(time, "time", _time)
    return t

@pytest.fixture
def dummy_telemetry():
    return DummyTelemetry()

@pytest.fixture
def dummy_memory_optimizer():
    return DummyMemoryOptimizer()

@pytest.fixture
def patched_globals(monkeypatch, dummy_telemetry, dummy_memory_optimizer):
    # AdaptiveResourceManager references get_global_telemetry and get_global_memory_optimizer indirectly at construction.
    # We patch those global getters in the module namespace to return our dummies.
    # If these getters are defined in the same module, they might be imported via "from x import get_global_telemetry".
    # We patch attributes on the imported module object directly.
    import adaptive_resource_manager as arm_mod  # type: ignore
    if hasattr(arm_mod, "get_global_telemetry"):
        monkeypatch.setattr(arm_mod, "get_global_telemetry", lambda: dummy_telemetry, raising=False)
    if hasattr(arm_mod, "get_global_memory_optimizer"):
        monkeypatch.setattr(arm_mod, "get_global_memory_optimizer", lambda: dummy_memory_optimizer, raising=False)
    return arm_mod

@pytest.fixture
def fake_psutil(monkeypatch):
    # Build a fake psutil with deterministic values
    class CPU:
        def __call__(self, interval=None):
            # Return 25% by default
            return 25.0

    class Mem:
        percent = 40.0

    class DiskIO:
        read_count = 100
        write_count = 200

    class NetIO:
        bytes_sent = 5 * 1024 * 1024
        bytes_recv = 3 * 1024 * 1024

    class FakePsutil:
        def cpu_percent(self, interval=None):
            return CPU()()

        def virtual_memory(self):
            return Mem()

        def disk_io_counters(self):
            return DiskIO()

        def net_io_counters(self):
            return NetIO()

    fake = FakePsutil()
    import adaptive_resource_manager as arm_mod  # type: ignore
    monkeypatch.setattr(arm_mod, "psutil", fake, raising=False)
    return fake

def test_resource_metrics_update_and_pressure_transitions(fake_time):
    m = ResourceMetrics(
        resource_type=ResourceType.CPU,
        current_usage=0.0,
        peak_usage=0.0,
        average_usage=0.0,
        pressure_level=PressureLevel.LOW,
    )
    # Update with low usage
    m.update_usage(0.2)
    assert m.current_usage == 0.2
    assert math.isclose(m.peak_usage, 0.2)
    assert m.pressure_level == PressureLevel.LOW
    assert len(m.usage_history) == 1
    # Update to moderate
    m.update_usage(0.65)
    assert m.pressure_level == PressureLevel.MODERATE
    assert m.peak_usage == 0.65
    # Update to high
    m.update_usage(0.81)
    assert m.pressure_level == PressureLevel.HIGH
    # Update to critical
    m.update_usage(0.98)
    assert m.pressure_level == PressureLevel.CRITICAL
    assert m.peak_usage == 0.98
    # Average should be mean of history
    assert math.isclose(m.average_usage, sum(m.usage_history) / len(m.usage_history), rel_tol=1e-9)

def test_resource_metrics_trend_detection():
    m = ResourceMetrics(
        resource_type=ResourceType.MEMORY,
        current_usage=0.0,
        peak_usage=0.0,
        average_usage=0.0,
        pressure_level=PressureLevel.LOW,
    )
    # Not enough history => stable
    assert m.get_trend() == "stable"
    # Fill history with stable pattern
    for _ in range(30):
        m.update_usage(0.5)
    assert m.get_trend() == "stable"
    # Increasing trend
    for i in range(20):
        m.update_usage(0.5 + i * 0.01)
    assert m.get_trend() in {"increasing", "stable"}  # depends on averaging windows
    # Decreasing trend
    for i in range(20):
        m.update_usage(0.8 - i * 0.02)
    tr = m.get_trend()
    assert tr in {"decreasing", "stable"}

def test_resource_limit_adaptation_bounds_and_thresholds():
    rl = ResourceLimit(
        resource_type=ResourceType.CPU,
        min_value=0.1,
        max_value=1.0,
        current_value=0.8,
        adaptation_rate=0.5,  # amplify effect for test
        stability_threshold=0.0,  # allow any change
    )
    # Critical pressure with aggressive strategy should reduce
    new_val = rl.adapt_to_pressure(PressureLevel.CRITICAL, AdaptationStrategy.AGGRESSIVE)
    assert new_val < 0.8
    assert rl.min_value <= new_val <= rl.max_value

    # Low pressure with conservative strategy should increase
    rl.current_value = 0.5
    new_val2 = rl.adapt_to_pressure(PressureLevel.LOW, AdaptationStrategy.CONSERVATIVE)
    assert new_val2 > 0.5
    assert rl.min_value <= new_val2 <= rl.max_value

    # Moderate pressure yields zero adjustment
    rl.current_value = 0.6
    new_val3 = rl.adapt_to_pressure(PressureLevel.MODERATE, AdaptationStrategy.MODERATE)
    assert new_val3 == 0.6

def test_resource_limit_stability_threshold_blocks_small_changes():
    rl = ResourceLimit(
        resource_type=ResourceType.MEMORY,
        min_value=0.1,
        max_value=0.9,
        current_value=0.5,
        adaptation_rate=0.01,       # very small effective change
        stability_threshold=0.05,   # requires >= 0.05 change
    )
    out = rl.adapt_to_pressure(PressureLevel.LOW, AdaptationStrategy.CONSERVATIVE)
    # Effective change should be below threshold, leaving current_value unchanged
    assert out == 0.5

def test_adaptation_event_change_percentage():
    ev = AdaptationEvent(
        timestamp=0.0,
        resource_type=ResourceType.CACHE,
        old_value=200.0,
        new_value=250.0,
        pressure_level=PressureLevel.HIGH,
        strategy=AdaptationStrategy.AGGRESSIVE,
        reason="test",
    )
    assert math.isclose(ev.change_percentage, 25.0)
    ev2 = AdaptationEvent(
        timestamp=0.0,
        resource_type=ResourceType.CACHE,
        old_value=0.0,
        new_value=10.0,
        pressure_level=PressureLevel.LOW,
        strategy=AdaptationStrategy.CONSERVATIVE,
        reason="test",
    )
    assert ev2.change_percentage == 0.0

@ASYNC_MARK
async def test_system_monitor_collects_metrics(fake_psutil):
    mon = SystemMonitor(monitoring_interval=0.0)
    # Call private collection once
    await mon._collect_system_metrics()
    metrics = mon.get_metrics()
    # Ensure metrics exists for resource types
    assert isinstance(metrics, dict)
    # At least CPU and MEMORY should be updated by fake_psutil
    cpu = mon.get_metric(ResourceType.CPU)
    mem = mon.get_metric(ResourceType.MEMORY)
    assert cpu is not None and 0.0 <= cpu.current_usage <= 1.0
    assert mem is not None and 0.0 <= mem.current_usage <= 1.0

@ASYNC_MARK
async def test_adaptive_resource_manager_perform_adaptation_low_pressure(patched_globals, fake_psutil, fake_time):
    # Low pressures should prefer conservative/moderate changes with telemetry recorded
    arm = AdaptiveResourceManager(
        monitoring_interval=0.0,
        adaptation_interval=0.01,
        default_strategy=AdaptationStrategy.MODERATE,
        enable_telemetry=True,
    )
    # Prime metrics by collecting once
    await arm._system_monitor._collect_system_metrics()
    # Perform adaptation once
    await arm._perform_adaptation()
    stats = arm.get_stats()
    # total_adaptations may be zero or more depending on stability thresholds; ensure stats keys present
    assert "adaptation_stats" in stats
    # Telemetry may have been recorded if any adaptation occurred
    # We check that no exceptions occur and the structure is valid
    hist = arm.get_adaptation_history()
    # Events in history (if any) should be AdaptationEvent
    for ev in hist:
        assert hasattr(ev, "resource_type")
        assert hasattr(ev, "strategy")

@ASYNC_MARK
async def test_adaptive_resource_manager_selects_strategy_based_on_pressure(patched_globals, fake_psutil):
    arm = AdaptiveResourceManager(enable_telemetry=False)
    # Build metrics with critical levels to force EMERGENCY
    metrics = arm.get_system_metrics()
    for m in metrics.values():
        m.pressure_level = PressureLevel.CRITICAL
    overall = arm._calculate_overall_pressure(metrics)
    assert overall in {PressureLevel.CRITICAL, PressureLevel.HIGH}
    strategy = arm._select_adaptation_strategy(overall, metrics)
    assert strategy == AdaptationStrategy.EMERGENCY

    # High with increasing trend => AGGRESSIVE
    for m in metrics.values():
        m.pressure_level = PressureLevel.HIGH
        # Craft history to produce increasing trend
        m.usage_history = deque([0.5] * 15 + [0.8] * 15, maxlen=100)
    strategy2 = arm._select_adaptation_strategy(PressureLevel.HIGH, metrics)
    assert strategy2 == AdaptationStrategy.AGGRESSIVE

def test_calculate_overall_pressure_weighting(patched_globals):
    arm = AdaptiveResourceManager(enable_telemetry=False)
    metrics = arm.get_system_metrics()
    # Force mixed pressures: 1 CRITICAL, others LOW
    first = True
    for m in metrics.values():
        if first:
            m.pressure_level = PressureLevel.CRITICAL
            first = False
        else:
            m.pressure_level = PressureLevel.LOW
    overall = arm._calculate_overall_pressure(metrics)
    # Depending on number of metrics, average may fall to MODERATE or HIGH
    assert overall in {PressureLevel.MODERATE, PressureLevel.HIGH, PressureLevel.CRITICAL}

@ASYNC_MARK
async def test_apply_adaptation_success_and_telemetry(patched_globals, fake_psutil, dummy_telemetry):
    arm = AdaptiveResourceManager(enable_telemetry=True)
    # Ensure a specific resource limit to adjust (CACHE)
    limit = arm.get_resource_limit(ResourceType.CACHE)
    assert limit is not None
    old = limit.current_value
    event = AdaptationEvent(
        timestamp=time.time(),
        resource_type=ResourceType.CACHE,
        old_value=old,
        new_value=old + 100,
        pressure_level=PressureLevel.HIGH,
        strategy=AdaptationStrategy.MODERATE,
        reason="test apply",
    )
    ok = await arm._apply_adaptation(event)
    assert ok is True
    # Telemetry recorded metric for resource limit
    assert any(name.startswith("resource_manager.") and "limit" in name for (name, _, _) in dummy_telemetry.metrics)

@ASYNC_MARK
async def test_apply_adaptation_handles_exception_and_records_failure(monkeypatch, patched_globals, dummy_telemetry):
    arm = AdaptiveResourceManager(enable_telemetry=True)

    # Force _update_cache_sizes to raise
    async def boom(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(arm, "_update_cache_sizes", boom, raising=True)

    limit = arm.get_resource_limit(ResourceType.CACHE)
    assert limit is not None
    event = AdaptationEvent(
        timestamp=time.time(),
        resource_type=ResourceType.CACHE,
        old_value=limit.current_value,
        new_value=limit.current_value + 1,
        pressure_level=PressureLevel.HIGH,
        strategy=AdaptationStrategy.MODERATE,
        reason="force failure",
    )
    ok = await arm._apply_adaptation(event)
    assert ok is False
    # Failure counter incremented with tags
    assert any(
        name == "resource_manager.adaptation_failures" and tags.get("resource_type") == "cache"
        for (name, _, tags) in dummy_telemetry.counters
    )

@ASYNC_MARK
async def test_optimize_memory_usage_calls_gc_when_high_pressure(monkeypatch, patched_globals, dummy_memory_optimizer):
    arm = AdaptiveResourceManager(enable_telemetry=False)
    # Spy on gc.collect
    calls = {"gc": 0}
    import gc
    def fake_collect():
        calls["gc"] += 1
        return 0
    monkeypatch.setattr(gc, "collect", fake_collect, raising=True)

    # target_usage < 0.5 triggers gc.collect
    await arm._optimize_memory_usage(0.3)
    assert calls["gc"] >= 1
    # Memory optimizer used
    assert any(name == "adaptive_memory_optimization" for (name, obj) in dummy_memory_optimizer.tracked)

def test_get_recommendations_reports_critical_and_increasing(patched_globals):
    arm = AdaptiveResourceManager(enable_telemetry=False)
    metrics = arm.get_system_metrics()
    # Force one critical
    cpu = metrics[ResourceType.CPU]
    cpu.current_usage = 0.99
    cpu.pressure_level = PressureLevel.CRITICAL
    # Force one high + increasing
    mem = metrics[ResourceType.MEMORY]
    mem.current_usage = 0.85
    mem.pressure_level = PressureLevel.HIGH
    mem.usage_history = deque([0.4] * 15 + [0.9] * 15, maxlen=100)
    recs = arm.get_recommendations()
    kinds = [r["type"] for r in recs]
    assert "critical_pressure" in kinds or "increasing_pressure" in kinds

def test_global_singleton_helpers(monkeypatch, patched_globals):
    # Ensure global getter returns singleton instance and can set limits
    m1 = get_global_adaptive_resource_manager()
    m2 = get_global_adaptive_resource_manager()
    assert m1 is m2
    # set_resource_limit should forward to manager
    set_resource_limit(ResourceType.CACHE, 100.0, 200.0, current_value=150.0)
    limit = m1.get_resource_limit(ResourceType.CACHE)
    assert limit is not None
    assert limit.min_value == 100.0
    assert limit.max_value == 200.0
    assert limit.current_value == 150.0

@ASYNC_MARK
async def test_start_stop_lifecycle(monkeypatch, patched_globals, fake_psutil):
    arm = AdaptiveResourceManager(monitoring_interval=0.0, adaptation_interval=0.01, enable_telemetry=False)
    await arm.start()
    # Give loops a moment to tick
    await asyncio.sleep(0.02)
    await arm.stop()
    # Repeated stops should be no-op
    await arm.stop()