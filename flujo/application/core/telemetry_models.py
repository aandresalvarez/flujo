"""
Telemetry data models and utilities.

This module provides PerformanceMetrics, StepAnalysis, ExecutionStats models,
CircularBuffer and BatchProcessor for efficient data handling, and metric
aggregation and reporting utilities for comprehensive telemetry data management.
"""

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple, TypeVar
import statistics
import uuid

from flujo.type_definitions.common import JSONObject

T = TypeVar("T")


class ExecutionStatus(Enum):
    """Execution status for telemetry."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRY = "retry"


class ResourceType(Enum):
    """Types of resources for monitoring."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for execution tracking."""

    # Timing metrics
    execution_time_ns: int = 0
    queue_time_ns: int = 0
    processing_time_ns: int = 0

    # Memory metrics
    memory_allocated_bytes: int = 0
    memory_peak_bytes: int = 0
    memory_freed_bytes: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0

    # Object pool metrics
    object_pool_hits: int = 0
    object_pool_misses: int = 0
    object_pool_creates: int = 0

    # Garbage collection metrics
    gc_collections: int = 0
    gc_time_ns: int = 0

    # Concurrency metrics
    concurrent_executions: int = 0
    semaphore_waits: int = 0
    semaphore_wait_time_ns: int = 0

    # Error metrics
    retry_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

    # Resource utilization
    cpu_time_ns: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0

    # Metadata
    timestamp_ns: int = field(default_factory=lambda: time.perf_counter_ns())
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.execution_time_ns / 1_000_000

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    @property
    def object_pool_hit_rate(self) -> float:
        """Calculate object pool hit rate."""
        total_requests = self.object_pool_hits + self.object_pool_misses
        return self.object_pool_hits / total_requests if total_requests > 0 else 0.0

    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency (freed/allocated)."""
        return self.memory_freed_bytes / max(self.memory_allocated_bytes, 1)

    def merge(self, other: "PerformanceMetrics") -> "PerformanceMetrics":
        """Merge with another PerformanceMetrics instance."""
        return PerformanceMetrics(
            execution_time_ns=self.execution_time_ns + other.execution_time_ns,
            queue_time_ns=self.queue_time_ns + other.queue_time_ns,
            processing_time_ns=self.processing_time_ns + other.processing_time_ns,
            memory_allocated_bytes=self.memory_allocated_bytes + other.memory_allocated_bytes,
            memory_peak_bytes=max(self.memory_peak_bytes, other.memory_peak_bytes),
            memory_freed_bytes=self.memory_freed_bytes + other.memory_freed_bytes,
            cache_hits=self.cache_hits + other.cache_hits,
            cache_misses=self.cache_misses + other.cache_misses,
            cache_evictions=self.cache_evictions + other.cache_evictions,
            object_pool_hits=self.object_pool_hits + other.object_pool_hits,
            object_pool_misses=self.object_pool_misses + other.object_pool_misses,
            object_pool_creates=self.object_pool_creates + other.object_pool_creates,
            gc_collections=self.gc_collections + other.gc_collections,
            gc_time_ns=self.gc_time_ns + other.gc_time_ns,
            concurrent_executions=max(self.concurrent_executions, other.concurrent_executions),
            semaphore_waits=self.semaphore_waits + other.semaphore_waits,
            semaphore_wait_time_ns=self.semaphore_wait_time_ns + other.semaphore_wait_time_ns,
            retry_count=self.retry_count + other.retry_count,
            error_count=self.error_count + other.error_count,
            timeout_count=self.timeout_count + other.timeout_count,
            cpu_time_ns=self.cpu_time_ns + other.cpu_time_ns,
            io_read_bytes=self.io_read_bytes + other.io_read_bytes,
            io_write_bytes=self.io_write_bytes + other.io_write_bytes,
            network_bytes_sent=self.network_bytes_sent + other.network_bytes_sent,
            network_bytes_received=self.network_bytes_received + other.network_bytes_received,
            timestamp_ns=min(self.timestamp_ns, other.timestamp_ns),
            execution_id=f"{self.execution_id}+{other.execution_id}",
        )


@dataclass
class StepAnalysis:
    """Pre-computed step analysis for optimization and telemetry."""

    # Basic identification
    step_name: str
    step_type: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Complexity analysis
    complexity_score: int = 1  # 1-10 scale
    estimated_duration_ms: float = 1.0
    estimated_memory_usage: int = 1024  # bytes
    estimated_cpu_usage: float = 0.1  # 0-1 scale

    # Feature analysis
    has_agent: bool = False
    has_processors: bool = False
    has_validators: bool = False
    has_plugins: bool = False
    has_fallback: bool = False
    has_cache: bool = False

    # Optimization flags
    can_use_fast_path: bool = False
    requires_deep_context_copy: bool = True
    supports_streaming: bool = False
    supports_parallel_execution: bool = False
    is_deterministic: bool = True
    is_cacheable: bool = True

    # Resource requirements
    required_resources: Set[ResourceType] = field(default_factory=set)
    optional_resources: Set[ResourceType] = field(default_factory=set)

    # Dependencies
    depends_on_context: bool = False
    depends_on_resources: bool = False
    depends_on_external_state: bool = False

    # Performance characteristics
    typical_execution_time_ms: float = 1.0
    typical_memory_usage_mb: float = 1.0
    typical_cache_hit_rate: float = 0.0

    # Analysis metadata
    analysis_timestamp: float = field(default_factory=time.time)
    analysis_version: str = "1.0"
    confidence_score: float = 1.0  # 0-1 scale

    @property
    def complexity_level(self) -> str:
        """Get human-readable complexity level."""
        if self.complexity_score <= 2:
            return "simple"
        elif self.complexity_score <= 5:
            return "moderate"
        elif self.complexity_score <= 8:
            return "complex"
        else:
            return "very_complex"

    @property
    def optimization_potential(self) -> float:
        """Calculate optimization potential (0-1 scale)."""
        potential = 0.0

        # Fast path potential
        if self.can_use_fast_path:
            potential += 0.3

        # Caching potential
        if self.is_cacheable and self.typical_cache_hit_rate < 0.5:
            potential += 0.2

        # Memory optimization potential
        if not self.requires_deep_context_copy:
            potential += 0.2

        # Parallel execution potential
        if self.supports_parallel_execution:
            potential += 0.2

        # Complexity reduction potential
        if self.complexity_score > 5:
            potential += 0.1

        return min(potential, 1.0)


@dataclass
class ExecutionStats:
    """Runtime execution statistics for performance monitoring."""

    # Execution counts
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    timeout_executions: int = 0

    # Timing statistics (in nanoseconds)
    total_execution_time_ns: int = 0
    min_execution_time_ns: int = 0  # Will be updated with actual min value
    max_execution_time_ns: int = 0

    # Memory statistics
    total_memory_allocated: int = 0
    peak_memory_usage: int = 0
    total_memory_freed: int = 0

    # Cache statistics
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_cache_evictions: int = 0

    # Concurrency statistics
    max_concurrent_executions: int = 0
    total_semaphore_waits: int = 0
    total_semaphore_wait_time_ns: int = 0

    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)
    last_error: Optional[str] = None
    last_error_timestamp: Optional[float] = None

    # Performance trends
    recent_execution_times: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    recent_memory_usage: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    recent_cache_hit_rates: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Metadata
    first_execution_timestamp: Optional[float] = None
    last_execution_timestamp: Optional[float] = None
    stats_version: str = "1.0"

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_executions == 0:
            return 0.0
        return self.failed_executions / self.total_executions

    @property
    def average_execution_time_ns(self) -> float:
        """Calculate average execution time."""
        if self.successful_executions == 0:
            return 0.0
        return self.total_execution_time_ns / self.successful_executions

    @property
    def average_execution_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        return self.average_execution_time_ns / 1_000_000

    @property
    def cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total_requests = self.total_cache_hits + self.total_cache_misses
        return self.total_cache_hits / total_requests if total_requests > 0 else 0.0

    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency."""
        return self.total_memory_freed / max(self.total_memory_allocated, 1)

    @property
    def execution_rate_per_second(self) -> float:
        """Calculate execution rate per second."""
        if not self.first_execution_timestamp or not self.last_execution_timestamp:
            return 0.0

        duration = self.last_execution_timestamp - self.first_execution_timestamp
        return self.total_executions / max(duration, 1.0)

    def update_with_metrics(self, metrics: PerformanceMetrics, status: ExecutionStatus) -> None:
        """Update statistics with new performance metrics."""
        # Update execution counts
        self.total_executions += 1

        if status == ExecutionStatus.SUCCESS:
            self.successful_executions += 1
        elif status == ExecutionStatus.FAILURE:
            self.failed_executions += 1
        elif status == ExecutionStatus.CANCELLED:
            self.cancelled_executions += 1
        elif status == ExecutionStatus.TIMEOUT:
            self.timeout_executions += 1

        # Update timing statistics
        if status == ExecutionStatus.SUCCESS:
            self.total_execution_time_ns += metrics.execution_time_ns
            if (
                self.min_execution_time_ns == 0
                or metrics.execution_time_ns < self.min_execution_time_ns
            ):
                self.min_execution_time_ns = metrics.execution_time_ns
            self.max_execution_time_ns = max(self.max_execution_time_ns, metrics.execution_time_ns)

            # Update recent trends
            self.recent_execution_times.append(metrics.execution_time_ns)

        # Update memory statistics
        self.total_memory_allocated += metrics.memory_allocated_bytes
        self.peak_memory_usage = max(self.peak_memory_usage, metrics.memory_peak_bytes)
        self.total_memory_freed += metrics.memory_freed_bytes
        self.recent_memory_usage.append(metrics.memory_peak_bytes)

        # Update cache statistics
        self.total_cache_hits += metrics.cache_hits
        self.total_cache_misses += metrics.cache_misses
        self.total_cache_evictions += metrics.cache_evictions

        if metrics.cache_hits + metrics.cache_misses > 0:
            hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
            self.recent_cache_hit_rates.append(hit_rate)

        # Update concurrency statistics
        self.max_concurrent_executions = max(
            self.max_concurrent_executions, metrics.concurrent_executions
        )
        self.total_semaphore_waits += metrics.semaphore_waits
        self.total_semaphore_wait_time_ns += metrics.semaphore_wait_time_ns

        # Update timestamps
        current_time = time.time()
        if self.first_execution_timestamp is None:
            self.first_execution_timestamp = current_time
        self.last_execution_timestamp = current_time

    def get_trend_analysis(self) -> JSONObject:
        """Analyze performance trends."""
        analysis = {}

        # Execution time trend
        if len(self.recent_execution_times) >= 10:
            times = [float(t) for t in self.recent_execution_times]
            analysis["execution_time_trend"] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "trend": self._calculate_trend(times),
            }

        # Memory usage trend
        if len(self.recent_memory_usage) >= 10:
            memory = [float(m) for m in self.recent_memory_usage]
            analysis["memory_usage_trend"] = {
                "mean": statistics.mean(memory),
                "median": statistics.median(memory),
                "stdev": statistics.stdev(memory) if len(memory) > 1 else 0,
                "trend": self._calculate_trend(memory),
            }

        # Cache hit rate trend
        if len(self.recent_cache_hit_rates) >= 10:
            hit_rates = list(self.recent_cache_hit_rates)
            analysis["cache_hit_rate_trend"] = {
                "mean": statistics.mean(hit_rates),
                "median": statistics.median(hit_rates),
                "stdev": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0,
                "trend": self._calculate_trend(hit_rates),
            }

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 3:
            return "insufficient_data"

        # Simple linear regression
        n = len(values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Determine trend based on slope
        if abs(slope) < statistics.stdev(values) * 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for reporting and analysis."""

    # Time window
    start_timestamp: float
    end_timestamp: float
    window_duration_seconds: float

    # Aggregated performance metrics
    total_executions: int = 0
    total_execution_time_ns: int = 0
    total_memory_allocated: int = 0
    total_cache_operations: int = 0

    # Statistical summaries
    execution_time_stats: Dict[str, float] = field(default_factory=dict)
    memory_usage_stats: Dict[str, float] = field(default_factory=dict)
    cache_performance_stats: Dict[str, float] = field(default_factory=dict)

    # Top performers and bottlenecks
    top_performers: List[Tuple[str, float]] = field(default_factory=list)
    bottlenecks: List[Tuple[str, float]] = field(default_factory=list)

    # Resource utilization
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)

    # Error summary
    error_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def executions_per_second(self) -> float:
        """Calculate executions per second."""
        return self.total_executions / max(self.window_duration_seconds, 1.0)

    @property
    def average_execution_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return (self.total_execution_time_ns / self.total_executions) / 1_000_000


class MetricAggregator:
    """Aggregates metrics for reporting and analysis."""

    def __init__(self, window_size_seconds: float = 300.0):  # 5 minutes default
        self.window_size_seconds = window_size_seconds
        self._metrics_buffer: deque[Tuple[float, str, PerformanceMetrics]] = deque()
        self._lock = RLock()

    def add_metrics(self, metrics: PerformanceMetrics, step_name: str) -> None:
        """Add metrics to aggregation buffer."""
        with self._lock:
            timestamp = time.time()
            self._metrics_buffer.append((timestamp, step_name, metrics))

            # Remove old metrics outside the window
            cutoff_time = timestamp - self.window_size_seconds
            while self._metrics_buffer and self._metrics_buffer[0][0] < cutoff_time:
                self._metrics_buffer.popleft()

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Get aggregated metrics for the current window."""
        with self._lock:
            if not self._metrics_buffer:
                current_time = time.time()
                return AggregatedMetrics(
                    start_timestamp=current_time,
                    end_timestamp=current_time,
                    window_duration_seconds=0.0,
                )

            # Calculate window bounds
            end_timestamp = time.time()
            start_timestamp = end_timestamp - self.window_size_seconds

            # Collect metrics within window
            window_metrics = []
            step_metrics = defaultdict(list)

            for timestamp, step_name, metrics in self._metrics_buffer:
                if timestamp >= start_timestamp:
                    window_metrics.append(metrics)
                    step_metrics[step_name].append(metrics)

            # Aggregate metrics
            aggregated = AggregatedMetrics(
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                window_duration_seconds=self.window_size_seconds,
            )

            if window_metrics:
                # Total aggregations
                aggregated.total_executions = len(window_metrics)
                aggregated.total_execution_time_ns = sum(
                    m.execution_time_ns for m in window_metrics
                )
                aggregated.total_memory_allocated = sum(
                    m.memory_allocated_bytes for m in window_metrics
                )
                aggregated.total_cache_operations = sum(
                    m.cache_hits + m.cache_misses for m in window_metrics
                )

                # Statistical summaries
                execution_times = [float(m.execution_time_ns) for m in window_metrics]
                memory_usage = [float(m.memory_peak_bytes) for m in window_metrics]
                cache_hit_rates = [
                    m.cache_hits / max(m.cache_hits + m.cache_misses, 1) for m in window_metrics
                ]

                aggregated.execution_time_stats = self._calculate_stats(execution_times)
                aggregated.memory_usage_stats = self._calculate_stats(memory_usage)
                aggregated.cache_performance_stats = self._calculate_stats(cache_hit_rates)

                # Top performers and bottlenecks
                step_performance = {}
                for step_name, metrics_list in step_metrics.items():
                    avg_time = sum(m.execution_time_ns for m in metrics_list) / len(metrics_list)
                    step_performance[step_name] = avg_time

                sorted_performance = sorted(step_performance.items(), key=lambda x: x[1])
                aggregated.top_performers = sorted_performance[:5]  # Top 5 fastest
                aggregated.bottlenecks = sorted_performance[-5:]  # Top 5 slowest

                # Error summary
                error_counts: Dict[str, int] = defaultdict(int)
                for metrics in window_metrics:
                    error_counts["total_errors"] += metrics.error_count
                    error_counts["total_timeouts"] += metrics.timeout_count
                    error_counts["total_retries"] += metrics.retry_count

                aggregated.error_summary = dict(error_counts)

            return aggregated

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical summary for values."""
        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p90": self._percentile(values, 90),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


class TelemetryReporter:
    """Generates reports from telemetry data."""

    def __init__(self) -> None:
        self._aggregator = MetricAggregator()

    def add_execution_metrics(self, step_name: str, metrics: PerformanceMetrics) -> None:
        """Add execution metrics for reporting."""
        self._aggregator.add_metrics(metrics, step_name)

    def generate_performance_report(self) -> JSONObject:
        """Generate comprehensive performance report."""
        aggregated = self._aggregator.get_aggregated_metrics()

        report = {
            "report_timestamp": time.time(),
            "window_duration_seconds": aggregated.window_duration_seconds,
            "summary": {
                "total_executions": aggregated.total_executions,
                "executions_per_second": aggregated.executions_per_second,
                "average_execution_time_ms": aggregated.average_execution_time_ms,
                "total_memory_allocated_mb": aggregated.total_memory_allocated / (1024 * 1024),
                "total_cache_operations": aggregated.total_cache_operations,
            },
            "performance_stats": {
                "execution_time": aggregated.execution_time_stats,
                "memory_usage": aggregated.memory_usage_stats,
                "cache_performance": aggregated.cache_performance_stats,
            },
            "top_performers": aggregated.top_performers,
            "bottlenecks": aggregated.bottlenecks,
            "errors": aggregated.error_summary,
        }

        return report

    def generate_trend_report(self, stats: ExecutionStats) -> JSONObject:
        """Generate trend analysis report."""
        trend_analysis = stats.get_trend_analysis()

        report = {
            "report_timestamp": time.time(),
            "overall_stats": {
                "total_executions": stats.total_executions,
                "success_rate": stats.success_rate,
                "failure_rate": stats.failure_rate,
                "average_execution_time_ms": stats.average_execution_time_ms,
                "cache_hit_rate": stats.cache_hit_rate,
                "memory_efficiency": stats.memory_efficiency,
                "execution_rate_per_second": stats.execution_rate_per_second,
            },
            "trends": trend_analysis,
            "error_breakdown": stats.error_types,
            "last_error": stats.last_error,
            "last_error_timestamp": stats.last_error_timestamp,
        }

        return report


# Global instances
_global_metric_aggregator: Optional[MetricAggregator] = None
_global_telemetry_reporter: Optional[TelemetryReporter] = None


def get_global_metric_aggregator() -> MetricAggregator:
    """Get the global metric aggregator instance."""
    global _global_metric_aggregator
    if _global_metric_aggregator is None:
        _global_metric_aggregator = MetricAggregator()
    return _global_metric_aggregator


def get_global_telemetry_reporter() -> TelemetryReporter:
    """Get the global telemetry reporter instance."""
    global _global_telemetry_reporter
    if _global_telemetry_reporter is None:
        _global_telemetry_reporter = TelemetryReporter()
    return _global_telemetry_reporter


# Convenience functions
def create_performance_metrics() -> PerformanceMetrics:
    """Create a new PerformanceMetrics instance."""
    return PerformanceMetrics()


def create_step_analysis(step_name: str, step_type: str) -> StepAnalysis:
    """Create a new StepAnalysis instance."""
    return StepAnalysis(step_name=step_name, step_type=step_type)


def create_execution_stats() -> ExecutionStats:
    """Create a new ExecutionStats instance."""
    return ExecutionStats()


def generate_performance_report() -> JSONObject:
    """Generate a performance report using the global reporter."""
    reporter = get_global_telemetry_reporter()
    return reporter.generate_performance_report()
