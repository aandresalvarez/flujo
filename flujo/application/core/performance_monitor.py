"""
Performance monitoring framework with real-time metrics.

This module provides real-time performance metrics collection, threshold detection,
regression alerts, statistical analysis, resource utilization tracking, and
bottleneck detection for comprehensive performance monitoring.
"""

import asyncio
import psutil
import time
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Callable, Tuple
from asyncio import Task
from threading import RLock

from .optimized_telemetry import get_global_telemetry, MetricType
from .optimization.memory.memory_utils import get_global_memory_optimizer
from .performance_models import (
    AlertLevel,
    MetricTrend,
    PerformanceAlert,
    PerformanceThreshold,
    SystemResourceMetrics,
    MetricStatistics,
)
from flujo.type_definitions.common import JSONObject


class PerformanceMonitor:
    """
    Real-time performance monitoring system.

    Features:
    - Real-time metric collection and analysis
    - Threshold-based alerting
    - Statistical analysis with trend detection
    - Resource utilization monitoring
    - Performance regression detection
    - Bottleneck identification
    """

    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 1000,
        enable_system_monitoring: bool = True,
        enable_alerts: bool = True,
        enable_stats: bool = True,
    ):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_alerts = enable_alerts
        self.enable_stats = enable_stats

        # Metric storage
        self._metric_history: Dict[str, deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._metric_stats: Dict[str, MetricStatistics] = {}

        # System resource tracking
        self._system_metrics_history: deque[SystemResourceMetrics] = deque(maxlen=history_size)
        self._last_disk_io: Optional[Tuple[float, Any]] = None
        self._last_network_io: Optional[Tuple[float, Any]] = None

        # Threshold management
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_history: List[PerformanceAlert] = []
        self._violation_counts: Dict[str, int] = defaultdict(int)

        # Alert callbacks
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        # Monitoring task
        self._monitoring_task: Optional[Task[Any]] = None
        self._shutdown_event = asyncio.Event()

        # Thread safety
        self._lock = RLock()

        # Integration with other systems
        self._telemetry = get_global_telemetry()
        self._memory_optimizer = get_global_memory_optimizer()

        # Performance tracking
        self._execution_times: deque[float] = deque(maxlen=100)
        self._memory_usage: deque[float] = deque(maxlen=100)

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self._monitoring_task is not None:
            return

        self._shutdown_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self._monitoring_task is None:
            return

        self._shutdown_event.set()
        await self._monitoring_task
        self._monitoring_task = None

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add performance threshold."""
        with self._lock:
            self._thresholds[threshold.metric_name] = threshold

    def remove_threshold(self, metric_name: str) -> None:
        """Remove performance threshold."""
        with self._lock:
            self._thresholds.pop(metric_name, None)

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add alert callback function."""
        with self._lock:
            self._alert_callbacks.append(callback)

    def record_metric(
        self, metric_name: str, value: float, timestamp: Optional[float] = None
    ) -> None:
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Store metric value
            self._metric_history[metric_name].append((timestamp, value))

            # Update statistics
            if self.enable_stats:
                self._update_metric_statistics(metric_name)

            # Check thresholds
            if self.enable_alerts and metric_name in self._thresholds:
                self._check_threshold(metric_name, value)

        # Record to telemetry
        self._telemetry.record_metric(
            f"performance.{metric_name}", value, MetricType.GAUGE, {"source": "performance_monitor"}
        )

    def get_metric_history(
        self, metric_name: str, window_seconds: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Get metric history."""
        with self._lock:
            history = list(self._metric_history[metric_name])

            if window_seconds is not None:
                cutoff_time = time.time() - window_seconds
                history = [(ts, val) for ts, val in history if ts >= cutoff_time]

            return history

    def get_metric_statistics(self, metric_name: str) -> Optional[MetricStatistics]:
        """Get metric statistics."""
        return self._metric_stats.get(metric_name)

    def get_current_alerts(self) -> List[PerformanceAlert]:
        """Get current active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, hours: Optional[float] = None) -> List[PerformanceAlert]:
        """Get alert history."""
        with self._lock:
            if hours is None:
                return list(self._alert_history)

            cutoff_time = time.time() - (hours * 3600)
            return [alert for alert in self._alert_history if alert.timestamp >= cutoff_time]

    def get_system_metrics(self) -> Optional[SystemResourceMetrics]:
        """Get latest system metrics."""
        with self._lock:
            if self._system_metrics_history:
                return self._system_metrics_history[-1]
            return None

    def get_system_metrics_history(
        self, window_seconds: Optional[float] = None
    ) -> List[SystemResourceMetrics]:
        """Get system metrics history."""
        with self._lock:
            history = list(self._system_metrics_history)

            if window_seconds is not None:
                cutoff_time = time.time() - window_seconds
                history = [metrics for metrics in history if metrics.timestamp >= cutoff_time]

            return history

    def detect_bottlenecks(self) -> JSONObject:
        """Detect performance bottlenecks."""
        bottlenecks: JSONObject = {}

        with self._lock:
            # Analyze system resources
            if self._system_metrics_history:
                latest_metrics = self._system_metrics_history[-1]

                if latest_metrics.cpu_percent > 80:
                    bottlenecks["cpu"] = {
                        "type": "cpu_bottleneck",
                        "severity": "high" if latest_metrics.cpu_percent > 90 else "medium",
                        "value": latest_metrics.cpu_percent,
                        "description": f"CPU usage at {latest_metrics.cpu_percent:.1f}%",
                    }

                if latest_metrics.memory_percent > 80:
                    bottlenecks["memory"] = {
                        "type": "memory_bottleneck",
                        "severity": "high" if latest_metrics.memory_percent > 90 else "medium",
                        "value": latest_metrics.memory_percent,
                        "description": f"Memory usage at {latest_metrics.memory_percent:.1f}%",
                    }

                if latest_metrics.disk_io_read_mb_s + latest_metrics.disk_io_write_mb_s > 100:
                    bottlenecks["disk_io"] = {
                        "type": "disk_io_bottleneck",
                        "severity": "medium",
                        "value": latest_metrics.disk_io_read_mb_s
                        + latest_metrics.disk_io_write_mb_s,
                        "description": f"High disk I/O: {latest_metrics.disk_io_read_mb_s + latest_metrics.disk_io_write_mb_s:.1f} MB/s",
                    }

            # Analyze metric trends
            for metric_name, stats in self._metric_stats.items():
                if stats.trend == MetricTrend.INCREASING and stats.trend_strength > 0.7:
                    if "latency" in metric_name.lower() or "duration" in metric_name.lower():
                        bottlenecks[f"trend_{metric_name}"] = {
                            "type": "performance_degradation",
                            "severity": "medium",
                            "value": stats.trend_strength,
                            "description": f"Increasing trend in {metric_name} (strength: {stats.trend_strength:.2f})",
                        }

        return bottlenecks

    def get_performance_summary(self) -> JSONObject:
        """Get comprehensive performance summary."""
        summary: JSONObject = {
            "timestamp": time.time(),
            "monitoring_active": self._monitoring_task is not None,
            "metrics_tracked": len(self._metric_history),
            "active_alerts": len(self._active_alerts),
            "total_alerts": len(self._alert_history),
        }

        # System metrics summary
        if self._system_metrics_history:
            latest_system = self._system_metrics_history[-1]
            summary["system"] = {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "process_memory_mb": latest_system.process_memory_mb,
                "process_threads": latest_system.process_threads,
            }

        # Top metrics by activity
        with self._lock:
            metric_activity = {name: len(history) for name, history in self._metric_history.items()}

            top_metrics = sorted(metric_activity.items(), key=lambda x: x[1], reverse=True)[:10]

            summary["top_metrics"] = top_metrics

        # Bottleneck summary
        bottlenecks = self.detect_bottlenecks()
        summary["bottlenecks"] = len(bottlenecks)
        summary["bottleneck_details"] = bottlenecks

        return summary

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                if self.enable_system_monitoring:
                    await self._collect_system_metrics()

                # Update statistics
                if self.enable_stats:
                    self._update_all_statistics()

                # Check for resolved alerts
                if self.enable_alerts:
                    self._check_resolved_alerts()

                # Sleep until next collection
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if collection fails
                continue

    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb_s = 0.0
            disk_write_mb_s = 0.0

            if self._last_disk_io and disk_io:
                time_delta = time.time() - self._last_disk_io[0]
                if time_delta > 0:
                    read_delta = disk_io.read_bytes - self._last_disk_io[1].read_bytes
                    write_delta = disk_io.write_bytes - self._last_disk_io[1].write_bytes
                    disk_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                    disk_write_mb_s = (write_delta / time_delta) / (1024 * 1024)

            if disk_io:
                self._last_disk_io = (time.time(), disk_io)

            # Get network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb_s = 0.0
            network_recv_mb_s = 0.0

            if self._last_network_io and network_io:
                time_delta = time.time() - self._last_network_io[0]
                if time_delta > 0:
                    sent_delta = network_io.bytes_sent - self._last_network_io[1].bytes_sent
                    recv_delta = network_io.bytes_recv - self._last_network_io[1].bytes_recv
                    network_sent_mb_s = (sent_delta / time_delta) / (1024 * 1024)
                    network_recv_mb_s = (recv_delta / time_delta) / (1024 * 1024)

            if network_io:
                self._last_network_io = (time.time(), network_io)

            # Get process-specific metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            process_threads = process.num_threads()

            try:
                process_files = len(process.open_files())
            except (psutil.AccessDenied, OSError):
                process_files = 0

            # Create metrics object
            metrics = SystemResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_io_read_mb_s=disk_read_mb_s,
                disk_io_write_mb_s=disk_write_mb_s,
                network_io_sent_mb_s=network_sent_mb_s,
                network_io_recv_mb_s=network_recv_mb_s,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory,
                process_threads=process_threads,
                process_open_files=process_files,
            )

            # Store metrics
            with self._lock:
                self._system_metrics_history.append(metrics)

            # Record individual metrics for threshold checking
            self.record_metric("system.cpu_percent", cpu_percent)
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric("process.cpu_percent", process_cpu)
            self.record_metric("process.memory_mb", process_memory)

        except Exception:
            # Silently ignore system metric collection errors
            pass

    def _update_metric_statistics(self, metric_name: str) -> None:
        """Update statistics for a specific metric."""
        history = self._metric_history[metric_name]
        if not history:
            return

        # Extract values from history
        values = [value for _, value in history]

        # Update statistics
        if metric_name not in self._metric_stats:
            self._metric_stats[metric_name] = MetricStatistics()

        self._metric_stats[metric_name].update(values)

    def _update_all_statistics(self) -> None:
        """Update statistics for all metrics."""
        with self._lock:
            for metric_name in self._metric_history:
                self._update_metric_statistics(metric_name)

    def _check_threshold(self, metric_name: str, value: float) -> None:
        """Check if metric value violates threshold."""
        threshold = self._thresholds[metric_name]
        violation_level = threshold.check_violation(value)

        if violation_level is not None:
            # Increment violation count
            self._violation_counts[metric_name] += 1

            # Check if we have enough consecutive violations
            if self._violation_counts[metric_name] >= threshold.consecutive_violations:
                self._trigger_alert(metric_name, violation_level, value, threshold)
        else:
            # Reset violation count
            self._violation_counts[metric_name] = 0

            # Check if alert should be resolved
            if metric_name in self._active_alerts:
                self._resolve_alert(metric_name)

    def _trigger_alert(
        self, metric_name: str, level: AlertLevel, value: float, threshold: PerformanceThreshold
    ) -> None:
        """Trigger performance alert."""
        # Get appropriate threshold value
        if level == AlertLevel.EMERGENCY and threshold.emergency_threshold is not None:
            threshold_value = threshold.emergency_threshold
        elif level == AlertLevel.CRITICAL:
            threshold_value = threshold.critical_threshold
        else:
            threshold_value = threshold.warning_threshold

        # Create alert
        alert = PerformanceAlert(
            metric_name=metric_name,
            level=level,
            current_value=value,
            threshold_value=threshold_value,
            message=f"{metric_name} {threshold.comparison_operator} {threshold_value} (current: {value})",
        )

        # Store alert
        self._active_alerts[metric_name] = alert
        self._alert_history.append(alert)

        # Reset violation count
        self._violation_counts[metric_name] = 0

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                # Continue with other callbacks
                continue

        # Record to telemetry
        self._telemetry.record_metric(
            "performance.alerts",
            1,
            MetricType.COUNTER,
            {"metric_name": metric_name, "level": level.value, "source": "performance_monitor"},
        )

    def _resolve_alert(self, metric_name: str) -> None:
        """Resolve active alert."""
        if metric_name not in self._active_alerts:
            return

        alert = self._active_alerts[metric_name]
        alert.resolved = True
        alert.resolution_timestamp = time.time()

        # Remove from active alerts
        del self._active_alerts[metric_name]

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                continue

    def _check_resolved_alerts(self) -> None:
        """Check if any alerts should be automatically resolved."""
        # This could be enhanced with more sophisticated resolution logic
        pass


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_global_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


# Convenience functions
def record_performance_metric(metric_name: str, value: float) -> None:
    """Convenience function to record performance metric."""
    monitor = get_global_performance_monitor()
    monitor.record_metric(metric_name, value)


def add_performance_threshold(
    metric_name: str,
    warning_threshold: float,
    critical_threshold: float,
    emergency_threshold: Optional[float] = None,
    comparison_operator: str = ">",
) -> None:
    """Convenience function to add performance threshold."""
    monitor = get_global_performance_monitor()
    threshold = PerformanceThreshold(
        metric_name=metric_name,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        emergency_threshold=emergency_threshold,
        comparison_operator=comparison_operator,
    )
    monitor.add_threshold(threshold)


def get_performance_summary() -> JSONObject:
    """Convenience function to get performance summary."""
    monitor = get_global_performance_monitor()
    return monitor.get_performance_summary()


def detect_performance_bottlenecks() -> JSONObject:
    """Convenience function to detect performance bottlenecks."""
    monitor = get_global_performance_monitor()
    return monitor.detect_bottlenecks()
