from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, Iterable, TypeVar, cast

from ..state.backends.base import StateBackend

T = TypeVar("T")


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` even if there's already a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Any = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        try:
            result = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover - unlikely
            exc = e

    thread = threading.Thread(target=_target)
    thread.start()
    thread.join()
    if exc:
        raise exc
    return cast(T, result)


try:
    from prometheus_client import start_http_server
    from prometheus_client.core import GaugeMetricFamily
    from prometheus_client.registry import REGISTRY

    PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False


class PrometheusCollector:
    """Prometheus collector that exposes aggregated run metrics."""

    def __init__(self, backend: StateBackend) -> None:
        if not PROM_AVAILABLE:
            raise ImportError("prometheus_client is not installed")
        self.backend = backend

    def collect(self) -> Iterable[GaugeMetricFamily]:
        stats: dict[str, Any] = run_coroutine(self.backend.get_workflow_stats())
        total = GaugeMetricFamily("flujo_runs_total", "Total pipeline runs")
        total.add_metric([], stats.get("total_workflows", 0))
        yield total

        status_counts = stats.get("status_counts", {})
        gauge = GaugeMetricFamily(
            "flujo_runs_by_status",
            "Pipeline runs by status",
            labels=["status"],
        )
        for status, count in status_counts.items():
            gauge.add_metric([status], count)
        yield gauge

        avg = GaugeMetricFamily(
            "flujo_avg_duration_ms",
            "Average pipeline duration in milliseconds",
        )
        avg.add_metric([], stats.get("average_execution_time_ms", 0))
        yield avg


def start_prometheus_server(port: int, backend: StateBackend) -> None:
    """Start a Prometheus metrics HTTP server in a daemon thread."""
    if not PROM_AVAILABLE:
        raise ImportError("prometheus_client is not installed")

    collector = PrometheusCollector(backend)
    if collector not in getattr(REGISTRY, "_collector_to_names", {}):
        REGISTRY.register(collector)

    def _run() -> None:
        start_http_server(port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
