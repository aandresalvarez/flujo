"""Telemetry integrations for production monitoring."""

from __future__ import annotations

from typing import Any

try:
    from .otel_hook import OpenTelemetryHook, StateBackendSpanExporter
except Exception:  # pragma: no cover - optional dependency

    class OpenTelemetryHook:  # type: ignore
        """Fallback when OpenTelemetry is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("OpenTelemetry extras are not installed")

    class StateBackendSpanExporter:  # type: ignore
        """Fallback when OpenTelemetry is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("OpenTelemetry extras are not installed")


try:
    from .prometheus import PrometheusCollector, start_prometheus_server
except Exception:  # pragma: no cover - optional dependency

    class PrometheusCollector:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Prometheus extras are not installed")

    def start_prometheus_server(*args: Any, **kwargs: Any) -> None:  # type: ignore
        raise ImportError("Prometheus extras are not installed")


__all__ = [
    "OpenTelemetryHook",
    "StateBackendSpanExporter",
    "PrometheusCollector",
    "start_prometheus_server",
]
