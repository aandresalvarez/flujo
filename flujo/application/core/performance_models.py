"""Shared performance monitoring models and enums."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AlertLevel(Enum):
    """Alert levels for performance monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricTrend(Enum):
    """Metric trend directions."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    comparison_operator: str = ">"  # >, <, ==, !=
    window_size: int = 10  # Number of samples to consider
    consecutive_violations: int = 3  # Consecutive violations before alert

    def check_violation(self, value: float) -> Optional[AlertLevel]:
        """Check if value violates threshold."""
        if self.emergency_threshold is not None:
            if self._compare(value, self.emergency_threshold):
                return AlertLevel.EMERGENCY

        if self._compare(value, self.critical_threshold):
            return AlertLevel.CRITICAL

        if self._compare(value, self.warning_threshold):
            return AlertLevel.WARNING

        return None

    def _compare(self, value: float, threshold: float) -> bool:
        """Compare value against threshold using operator."""
        if self.comparison_operator == ">":
            return value > threshold
        if self.comparison_operator == "<":
            return value < threshold
        if self.comparison_operator == "==":
            return abs(value - threshold) < 1e-10
        if self.comparison_operator == "!=":
            return abs(value - threshold) >= 1e-10
        return False


@dataclass
class PerformanceAlert:
    """Performance alert information."""

    metric_name: str
    level: AlertLevel
    current_value: float
    threshold_value: float
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get alert duration in seconds."""
        if not self.resolved or self.resolution_timestamp is None:
            return None
        return self.resolution_timestamp - self.timestamp


@dataclass
class MetricStatistics:
    """Statistical analysis of metric values."""

    count: int = 0
    sum: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    mean: float = 0.0
    variance: float = 0.0
    std_dev: float = 0.0

    # Percentiles
    p50: float = 0.0  # Median
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    # Trend analysis
    trend: MetricTrend = MetricTrend.STABLE
    trend_strength: float = 0.0  # 0-1, strength of trend

    def update(self, values: List[float]) -> None:
        """Update statistics with new values."""
        if not values:
            return

        self.count = len(values)
        self.sum = sum(values)
        self.min_value = min(values)
        self.max_value = max(values)
        self.mean = self.sum / self.count

        if self.count > 1:
            self.variance = statistics.variance(values)
            self.std_dev = statistics.stdev(values)

        # Calculate percentiles
        sorted_values = sorted(values)
        self.p50 = self._percentile(sorted_values, 50)
        self.p90 = self._percentile(sorted_values, 90)
        self.p95 = self._percentile(sorted_values, 95)
        self.p99 = self._percentile(sorted_values, 99)

        # Analyze trend
        self._analyze_trend(values)

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not sorted_values:
            return 0.0

        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def _analyze_trend(self, values: List[float]) -> None:
        """Analyze trend in values."""
        if len(values) < 3:
            self.trend = MetricTrend.STABLE
            self.trend_strength = 0.0
            return

        # Simple linear regression to detect trend
        n = len(values)
        x_values = list(range(n))

        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Determine trend
        slope_threshold = self.std_dev * 0.1  # 10% of standard deviation

        if abs(slope) < slope_threshold:
            self.trend = MetricTrend.STABLE
        elif slope > slope_threshold:
            self.trend = MetricTrend.INCREASING
        else:
            self.trend = MetricTrend.DECREASING

        # Calculate trend strength (correlation coefficient)
        if denominator > 0 and self.std_dev > 0:
            correlation = abs(numerator) / (n * self.std_dev * statistics.stdev(x_values))
            self.trend_strength = min(correlation, 1.0)
        else:
            self.trend_strength = 0.0

        # Check for volatility
        if self.std_dev > self.mean * 0.5:  # High relative standard deviation
            self.trend = MetricTrend.VOLATILE


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    disk_io_read_mb_s: float = 0.0
    disk_io_write_mb_s: float = 0.0
    network_io_sent_mb_s: float = 0.0
    network_io_recv_mb_s: float = 0.0

    # Process-specific metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_threads: int = 0
    process_open_files: int = 0

    timestamp: float = field(default_factory=time.time)


__all__ = [
    "AlertLevel",
    "MetricTrend",
    "PerformanceThreshold",
    "PerformanceAlert",
    "MetricStatistics",
    "SystemResourceMetrics",
]
