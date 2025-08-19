from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Coroutine
from typing import Any, Callable, Iterable, TypeVar, cast

from ..state.backends.base import StateBackend


class PrometheusBindingError(PermissionError):
    """Raised when the metrics server cannot bind due to environment restrictions."""

    pass


T = TypeVar("T")

# Default timeout for server readiness checks
DEFAULT_SERVER_TIMEOUT = 10.0


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


def _wait_for_server(host: str, port: int, timeout: float = DEFAULT_SERVER_TIMEOUT) -> bool:
    """Wait for a server to be ready to accept connections."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


try:
    from prometheus_client import start_http_server
    from prometheus_client.core import GaugeMetricFamily
    from prometheus_client.registry import REGISTRY

    PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False

    class Collector:
        """Stub class for Collector when prometheus_client is unavailable."""

        pass


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


def start_prometheus_server(port: int, backend: StateBackend) -> tuple[Callable[[], bool], int]:
    """Start a Prometheus metrics HTTP server in a daemon thread.

    Returns:
        A tuple of (wait_for_ready_function, assigned_port) where the function waits for the server to be ready and returns True if successful.
    """
    if not PROM_AVAILABLE:
        raise ImportError("prometheus_client is not installed")

    collector = PrometheusCollector(backend)
    # Register collector, ignore if already registered
    try:
        from typing import TYPE_CHECKING, cast

        if TYPE_CHECKING:
            # Precise type for register at type-check time
            from prometheus_client.registry import Collector as _PromCollector

            REGISTRY.register(cast(_PromCollector, collector))
        else:  # runtime path
            REGISTRY.register(collector)
    except ValueError:
        # Already registered, ignore
        pass

    # For port 0 we need to find an available port first
    if port == 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Bind explicitly to localhost to satisfy sandbox constraints
            sock.bind(("127.0.0.1", 0))
            assigned_port = sock.getsockname()[1]
        except PermissionError as e:  # pragma: no cover - sandbox-specific
            # Surface a domain error; tests can decide to skip in constrained envs
            raise PrometheusBindingError(f"Prometheus server binding not permitted: {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass
    else:
        assigned_port = port

    def _run() -> None:
        start_http_server(assigned_port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def wait_for_ready() -> bool:
        """Wait for the server to be ready to accept connections."""
        return _wait_for_server("localhost", assigned_port)

    return wait_for_ready, assigned_port
