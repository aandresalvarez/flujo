from __future__ import annotations

from flujo.application.runner_components import TracingManager
from flujo.tracing.manager import TraceManager, get_active_trace_manager


def test_tracing_manager_setup_enables_tracing(monkeypatch) -> None:
    manager = TracingManager(enable_tracing=True)
    hooks = manager.setup([])

    trace_manager = manager.trace_manager
    assert isinstance(trace_manager, TraceManager)
    assert trace_manager.hook in hooks
    assert get_active_trace_manager() is trace_manager

    manager.teardown()
    assert get_active_trace_manager() is None


def test_tracing_manager_respects_disable_env_and_attaches_console(monkeypatch) -> None:
    monkeypatch.setenv("FLUJO_DISABLE_TRACING", "1")

    manager = TracingManager(enable_tracing=True, local_tracer="default")
    hooks = manager.setup([])

    assert manager.trace_manager is None
    assert hooks  # console tracer hook should still be attached
    assert get_active_trace_manager() is None

    manager.teardown()
