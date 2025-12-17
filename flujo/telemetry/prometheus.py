from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any, Callable, Iterable, TypeVar

from ..state.backends.base import StateBackend
from ..utils.async_bridge import run_sync


class PrometheusBindingError(PermissionError):
    """Raised when the metrics server cannot bind due to environment restrictions."""

    pass


T = TypeVar("T")

# Default timeout for server readiness checks
DEFAULT_SERVER_TIMEOUT = 10.0

# Backward compatibility alias - use run_sync directly for new code
run_coroutine = run_sync


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
    from prometheus_client import make_wsgi_app
    from prometheus_client.core import GaugeMetricFamily
    from prometheus_client.registry import Collector, REGISTRY

    PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False

    class Collector:
        """Stub class for Collector when prometheus_client is unavailable."""

        pass


class PrometheusCollector(Collector):
    """Prometheus collector that exposes aggregated run metrics."""

    def __init__(self, backend: StateBackend) -> None:
        if not PROM_AVAILABLE:
            raise ImportError("prometheus_client is not installed")
        self.backend = backend

    def collect(self) -> Iterable[GaugeMetricFamily]:
        stats: dict[str, Any] = {}
        in_running_loop = False
        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if in_running_loop:
            get_sync = getattr(self.backend, "get_workflow_stats_sync", None)
            if callable(get_sync):
                try:
                    stats = get_sync()
                except Exception:
                    stats = {}
        else:
            try:
                stats = run_coroutine(self.backend.get_workflow_stats())
            except Exception:
                stats = {}
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

        # Background tasks by status
        bg_counts = stats.get("background_status_counts", {}) or {}
        if isinstance(bg_counts, dict):
            bg_gauge = GaugeMetricFamily(
                "flujo_background_tasks_by_status",
                "Background tasks by status",
                labels=["status"],
            )
            for status, count in bg_counts.items():
                bg_gauge.add_metric([status], count)
            yield bg_gauge


_SERVERS: list[tuple[Any, threading.Thread, int]] = []


def start_prometheus_server(port: int, backend: StateBackend) -> tuple[Callable[[], bool], int]:
    """Start a Prometheus metrics HTTP server in a controllable daemon thread.

    This implementation avoids relying on prometheus_client.start_http_server so we can
    fully control the server lifecycle (shutdown at exit) and guarantee daemonized
    threads. This prevents test runs from hanging due to lingering non-daemon threads.

    Returns:
        A tuple of (wait_for_ready_function, assigned_port) where the function waits
        for the server to be ready and returns True if successful.
    """
    if not PROM_AVAILABLE:
        raise ImportError("prometheus_client is not installed")

    collector = PrometheusCollector(backend)
    # Register collector, ignore if already registered
    try:
        # Collector is an ABC; we register our concrete implementation.
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

    # Build a small WSGI server hosting Prometheus' WSGI app so we can control it
    from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
    from socketserver import ThreadingMixIn
    import atexit

    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        # Ensure request handler threads do not block interpreter shutdown
        daemon_threads = True

    app = make_wsgi_app()

    # Bind explicitly to localhost
    httpd = make_server(
        "127.0.0.1",
        assigned_port,
        app,
        server_class=ThreadingWSGIServer,
        handler_class=WSGIRequestHandler,
    )

    def _serve_forever() -> None:  # pragma: no cover - exercised in integration
        try:
            httpd.serve_forever()
        except Exception:
            # Server is being shut down or environment is tearing down
            pass

    # Start serving in a daemonized thread under our control
    server_thread = threading.Thread(
        target=_serve_forever, name="flujo-prometheus-server", daemon=True
    )
    server_thread.start()

    # Ensure clean shutdown on interpreter exit to avoid hanging the test runner
    try:
        _SERVERS.append((httpd, server_thread, assigned_port))
        atexit.register(shutdown_all_prometheus_servers)
    except Exception:
        pass

    def wait_for_ready() -> bool:
        """Wait for the server to be ready to accept connections."""
        return _wait_for_server("localhost", assigned_port)

    return wait_for_ready, assigned_port


def _safe_shutdown(httpd: Any, server_thread: threading.Thread) -> None:
    """Attempt to gracefully stop the WSGI server and join its thread.

    This function is registered via atexit to avoid leaving non-daemon threads
    alive at interpreter shutdown, which can cause `threading._shutdown` hangs.
    """
    try:
        # Stop serving and close the listening socket
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass
    try:
        # Join quickly; thread is daemon so this is best-effort
        server_thread.join(timeout=1.0)
    except Exception:
        pass


def shutdown_all_prometheus_servers() -> None:
    """Shutdown all Prometheus servers started via start_prometheus_server.

    Useful for test environments to ensure no background servers keep the
    interpreter alive. Also registered via atexit for extra safety.
    """
    try:
        while _SERVERS:
            httpd, th, _port = _SERVERS.pop()
            _safe_shutdown(httpd, th)
    except Exception:
        # Best-effort cleanup
        pass
