"""
Adaptive resource manager with dynamic adjustment capabilities.

This module provides dynamic resource adjustment, memory pressure detection,
automatic cache size tuning, CPU utilization monitoring, and concurrency
limit adjustment for optimal system performance under varying loads.
"""

import asyncio
import time
import psutil
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from asyncio import Task
from threading import RLock
from enum import Enum
import statistics
import multiprocessing

from .optimized_telemetry import get_global_telemetry
from .optimization.memory.memory_utils import get_global_memory_optimizer
from flujo.type_definitions.common import JSONObject


class ResourceType(Enum):
    """Types of resources managed by the adaptive resource manager."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CONCURRENCY = "concurrency"
    CACHE = "cache"


class PressureLevel(Enum):
    """Resource pressure levels."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AdaptationStrategy(Enum):
    """Resource adaptation strategies."""

    CONSERVATIVE = "conservative"  # Small, gradual changes
    MODERATE = "moderate"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Large, rapid changes
    EMERGENCY = "emergency"  # Immediate drastic changes


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    pressure_level: PressureLevel

    # Historical data
    usage_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    pressure_events: int = 0
    last_pressure_time: Optional[float] = None

    # Thresholds
    low_threshold: float = 0.3
    moderate_threshold: float = 0.6
    high_threshold: float = 0.8
    critical_threshold: float = 0.95

    def update_usage(self, usage: float) -> None:
        """Update resource usage metrics."""
        self.current_usage = usage
        self.peak_usage = max(self.peak_usage, usage)
        self.usage_history.append(usage)

        # Calculate average
        if self.usage_history:
            self.average_usage = statistics.mean(self.usage_history)

        # Update pressure level
        old_pressure = self.pressure_level
        self.pressure_level = self._calculate_pressure_level(usage)

        # Track pressure events
        if self.pressure_level.value > old_pressure.value:
            self.pressure_events += 1
            self.last_pressure_time = time.time()

    def _calculate_pressure_level(self, usage: float) -> PressureLevel:
        """Calculate pressure level based on usage."""
        if usage >= self.critical_threshold:
            return PressureLevel.CRITICAL
        elif usage >= self.high_threshold:
            return PressureLevel.HIGH
        elif usage >= self.moderate_threshold:
            return PressureLevel.MODERATE
        else:
            return PressureLevel.LOW

    def get_trend(self, window_size: int = 20) -> str:
        """Get usage trend over recent history."""
        if len(self.usage_history) < window_size:
            return "stable"

        recent = list(self.usage_history)[-window_size:]
        if len(recent) < 3:
            return "stable"

        # Simple trend analysis
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        diff = (second_avg - first_avg) / max(first_avg, 0.01)

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"


@dataclass
class ResourceLimit:
    """Resource limit configuration."""

    resource_type: ResourceType
    min_value: float
    max_value: float
    current_value: float
    target_value: Optional[float] = None

    # Adaptation parameters
    adaptation_rate: float = 0.1  # How quickly to adapt (0.0-1.0)
    stability_threshold: float = 0.05  # Minimum change threshold

    def adapt_to_pressure(
        self, pressure_level: PressureLevel, strategy: AdaptationStrategy
    ) -> float:
        """Adapt resource limit based on pressure level and strategy."""
        # old_value = self.current_value  # Unused variable

        # Calculate adaptation factor based on strategy
        if strategy == AdaptationStrategy.CONSERVATIVE:
            factor = 0.05
        elif strategy == AdaptationStrategy.MODERATE:
            factor = 0.1
        elif strategy == AdaptationStrategy.AGGRESSIVE:
            factor = 0.2
        else:  # EMERGENCY
            factor = 0.5

        # Adjust based on pressure level
        if pressure_level == PressureLevel.CRITICAL:
            # Reduce resource limit significantly
            adjustment = -factor * self.current_value
        elif pressure_level == PressureLevel.HIGH:
            # Reduce resource limit moderately
            adjustment = -factor * 0.5 * self.current_value
        elif pressure_level == PressureLevel.LOW:
            # Increase resource limit if possible
            adjustment = factor * 0.3 * self.current_value
        else:  # MODERATE
            # Small adjustment based on trend
            adjustment = 0

        # Apply adaptation rate
        adjustment *= self.adaptation_rate

        # Calculate new value
        new_value = self.current_value + adjustment

        # Ensure within bounds
        new_value = max(self.min_value, min(self.max_value, new_value))

        # Apply stability threshold
        if abs(new_value - self.current_value) >= self.stability_threshold:
            self.current_value = new_value

        return self.current_value


@dataclass
class AdaptationEvent:
    """Record of a resource adaptation event."""

    timestamp: float
    resource_type: ResourceType
    old_value: float
    new_value: float
    pressure_level: PressureLevel
    strategy: AdaptationStrategy
    reason: str
    success: bool = True

    @property
    def change_percentage(self) -> float:
        """Calculate percentage change."""
        if self.old_value == 0:
            return 0.0
        return ((self.new_value - self.old_value) / self.old_value) * 100


class SystemMonitor:
    """System resource monitoring component."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self._monitoring_task: Optional[Task[Any]] = None
        self._optimization_task: Optional[Task[Any]] = None
        self._shutdown_event = asyncio.Event()
        self._metrics: Dict[ResourceType, ResourceMetrics] = {}
        self._lock = RLock()

        # Initialize metrics
        for resource_type in ResourceType:
            self._metrics[resource_type] = ResourceMetrics(
                resource_type=resource_type,
                current_usage=0.0,
                peak_usage=0.0,
                average_usage=0.0,
                pressure_level=PressureLevel.LOW,
            )

    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self._monitoring_task is not None:
            return

        self._shutdown_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if self._monitoring_task is None:
            return

        self._shutdown_event.set()
        await self._monitoring_task
        self._monitoring_task = None

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if collection fails
                continue

    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self._update_metric(ResourceType.CPU, cpu_percent / 100.0)

            # Memory metrics
            memory = psutil.virtual_memory()
            self._update_metric(ResourceType.MEMORY, memory.percent / 100.0)

            # Disk I/O metrics
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Use a simple heuristic based on I/O operations
                    io_rate = (disk_io.read_count + disk_io.write_count) / 1000.0
                    io_usage = min(io_rate / 100.0, 1.0)  # Normalize to 0-1
                    self._update_metric(ResourceType.DISK_IO, io_usage)
            except (AttributeError, OSError):
                pass

            # Network I/O metrics
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    # Use a simple heuristic based on bytes transferred
                    net_rate = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)  # MB
                    net_usage = min(net_rate / 100.0, 1.0)  # Normalize to 0-1
                    self._update_metric(ResourceType.NETWORK_IO, net_usage)
            except (AttributeError, OSError):
                pass

        except Exception:
            # Continue monitoring even if some metrics fail
            pass

    def _update_metric(self, resource_type: ResourceType, usage: float) -> None:
        """Update resource metric."""
        with self._lock:
            if resource_type in self._metrics:
                self._metrics[resource_type].update_usage(usage)

    def get_metrics(self) -> Dict[ResourceType, ResourceMetrics]:
        """Get current resource metrics."""
        with self._lock:
            return self._metrics.copy()

    def get_metric(self, resource_type: ResourceType) -> Optional[ResourceMetrics]:
        """Get specific resource metric."""
        with self._lock:
            return self._metrics.get(resource_type)


class AdaptiveResourceManager:
    """
    Adaptive resource manager with dynamic adjustment capabilities.

    Features:
    - Dynamic resource adjustment based on system load
    - Memory pressure detection and automatic cache size tuning
    - CPU utilization monitoring and concurrency limit adjustment
    - Intelligent adaptation strategies
    - Resource limit enforcement
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        monitoring_interval: float = 0.5,
        adaptation_interval: float = 3.0,
        default_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE,
        enable_telemetry: bool = True,
    ):
        self.monitoring_interval = monitoring_interval
        self.adaptation_interval = adaptation_interval
        self.default_strategy = default_strategy
        self.enable_telemetry = enable_telemetry

        # Core components
        self._system_monitor = SystemMonitor(monitoring_interval)
        self._telemetry = get_global_telemetry() if enable_telemetry else None
        self._memory_optimizer = get_global_memory_optimizer()

        # Resource limits
        self._resource_limits: dict[ResourceType, ResourceLimit] = {}
        self._initialize_default_limits()

        # Adaptation tracking
        self._adaptation_history: deque[AdaptationEvent] = deque(maxlen=1000)
        self._last_adaptation_time = 0.0
        self._adaptation_task: Optional[Task[Any]] = None

        # Callbacks
        self._adaptation_callbacks: List[Callable[[AdaptationEvent], None]] = []

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._stats: JSONObject = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "failed_adaptations": 0,
            "pressure_events": defaultdict(int),
            "adaptation_events": defaultdict(int),
        }

        # Performance tracking
        self._resource_usage: deque[float] = deque(maxlen=100)
        self._allocation_history: deque[JSONObject] = deque(maxlen=100)

    def _initialize_default_limits(self) -> None:
        """Initialize default resource limits."""
        cpu_count = multiprocessing.cpu_count()

        self._resource_limits = {
            ResourceType.CPU: ResourceLimit(
                resource_type=ResourceType.CPU, min_value=0.1, max_value=1.0, current_value=0.8
            ),
            ResourceType.MEMORY: ResourceLimit(
                resource_type=ResourceType.MEMORY, min_value=0.1, max_value=0.9, current_value=0.7
            ),
            ResourceType.CONCURRENCY: ResourceLimit(
                resource_type=ResourceType.CONCURRENCY,
                min_value=1.0,
                max_value=cpu_count * 4,
                current_value=cpu_count * 2,
            ),
            ResourceType.CACHE: ResourceLimit(
                resource_type=ResourceType.CACHE,
                min_value=100.0,
                max_value=10000.0,
                current_value=1000.0,
            ),
        }

    async def start(self) -> None:
        """Start adaptive resource management."""
        await self._system_monitor.start_monitoring()

        if self._adaptation_task is None:
            self._adaptation_task = asyncio.create_task(self._adaptation_loop())

    async def stop(self) -> None:
        """Stop adaptive resource management."""
        await self._system_monitor.stop_monitoring()

        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
            self._adaptation_task = None

    # Async context manager to guarantee cleanup
    async def __aenter__(self) -> "AdaptiveResourceManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()

    async def _adaptation_loop(self) -> None:
        """Main adaptation loop."""
        while True:
            try:
                await asyncio.sleep(self.adaptation_interval)
                await self._perform_adaptation()
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue adaptation even if one cycle fails
                continue

    async def _perform_adaptation(self) -> None:
        """Perform resource adaptation based on current metrics."""
        current_time = time.time()

        # Get current system metrics
        metrics = self._system_monitor.get_metrics()

        # Determine overall system pressure
        overall_pressure = self._calculate_overall_pressure(metrics)

        # Select adaptation strategy
        strategy = self._select_adaptation_strategy(overall_pressure, metrics)

        # Adapt each resource type
        adaptations = []

        for resource_type, limit in self._resource_limits.items():
            if resource_type in metrics:
                metric = metrics[resource_type]
                old_value = limit.current_value

                # Adapt the limit
                new_value = limit.adapt_to_pressure(metric.pressure_level, strategy)

                if new_value != old_value:
                    # Create adaptation event
                    event = AdaptationEvent(
                        timestamp=current_time,
                        resource_type=resource_type,
                        old_value=old_value,
                        new_value=new_value,
                        pressure_level=metric.pressure_level,
                        strategy=strategy,
                        reason=f"Pressure: {metric.pressure_level.value}, Trend: {metric.get_trend()}",
                    )

                    # Apply the adaptation
                    success = await self._apply_adaptation(event)
                    event.success = success

                    adaptations.append(event)

                    # Record adaptation
                    with self._lock:
                        self._adaptation_history.append(event)
                        self._stats["total_adaptations"] += 1
                        if success:
                            self._stats["successful_adaptations"] += 1
                        else:
                            self._stats["failed_adaptations"] += 1

                        self._stats["adaptation_events"][resource_type] += 1

        # Notify callbacks
        for event in adaptations:
            for callback in self._adaptation_callbacks:
                try:
                    callback(event)
                except Exception:
                    continue

        # Record telemetry
        if self._telemetry and adaptations:
            self._telemetry.record_metric(
                "resource_manager.adaptations", len(adaptations), tags={"strategy": strategy.value}
            )

        self._last_adaptation_time = current_time

    def _calculate_overall_pressure(
        self, metrics: Dict[ResourceType, ResourceMetrics]
    ) -> PressureLevel:
        """Calculate overall system pressure level."""
        pressure_scores = {
            PressureLevel.LOW: 0,
            PressureLevel.MODERATE: 1,
            PressureLevel.HIGH: 2,
            PressureLevel.CRITICAL: 3,
        }

        total_score = 0
        count = 0

        for metric in metrics.values():
            total_score += pressure_scores[metric.pressure_level]
            count += 1

        if count == 0:
            return PressureLevel.LOW

        avg_score = total_score / count

        if avg_score >= 2.5:
            return PressureLevel.CRITICAL
        elif avg_score >= 1.5:
            return PressureLevel.HIGH
        elif avg_score >= 0.5:
            return PressureLevel.MODERATE
        else:
            return PressureLevel.LOW

    def _select_adaptation_strategy(
        self, overall_pressure: PressureLevel, metrics: Dict[ResourceType, ResourceMetrics]
    ) -> AdaptationStrategy:
        """Select appropriate adaptation strategy."""
        # Check for critical conditions
        critical_resources = [
            m for m in metrics.values() if m.pressure_level == PressureLevel.CRITICAL
        ]

        if critical_resources:
            return AdaptationStrategy.EMERGENCY

        # Check for high pressure with increasing trend
        high_pressure_increasing = [
            m
            for m in metrics.values()
            if m.pressure_level == PressureLevel.HIGH and m.get_trend() == "increasing"
        ]

        if high_pressure_increasing:
            return AdaptationStrategy.AGGRESSIVE

        # Use overall pressure level
        if overall_pressure == PressureLevel.HIGH:
            return AdaptationStrategy.AGGRESSIVE
        elif overall_pressure == PressureLevel.MODERATE:
            return AdaptationStrategy.MODERATE
        else:
            return AdaptationStrategy.CONSERVATIVE

    async def _apply_adaptation(self, event: AdaptationEvent) -> bool:
        """Apply resource adaptation."""
        try:
            if event.resource_type == ResourceType.CONCURRENCY:
                # Update concurrency limits in the system
                await self._update_concurrency_limit(int(event.new_value))

            elif event.resource_type == ResourceType.CACHE:
                # Update cache sizes
                await self._update_cache_sizes(int(event.new_value))

            elif event.resource_type == ResourceType.MEMORY:
                # Trigger memory optimization
                await self._optimize_memory_usage(event.new_value)

            # Record successful adaptation
            if self._telemetry:
                self._telemetry.record_metric(
                    f"resource_manager.{event.resource_type.value}_limit",
                    event.new_value,
                    tags={"strategy": event.strategy.value},
                )

            return True

        except Exception as e:
            # Record failed adaptation
            if self._telemetry:
                self._telemetry.increment_counter(
                    "resource_manager.adaptation_failures",
                    tags={"resource_type": event.resource_type.value, "error": type(e).__name__},
                )
            return False

    async def _update_concurrency_limit(self, new_limit: int) -> None:
        """Update system concurrency limits."""
        # This would integrate with the concurrency optimizer
        try:
            from .optimization.performance.concurrency import get_global_concurrency_optimizer

            concurrency_optimizer = get_global_concurrency_optimizer()

            # Update semaphore limits if available
            if hasattr(concurrency_optimizer, "_semaphore"):
                # Note: This is a simplified approach
                # In practice, you'd need more sophisticated semaphore updating
                pass
        except ImportError:
            pass

    async def _update_cache_sizes(self, new_size: int) -> None:
        """Update cache sizes across the system."""
        # This would integrate with various cache systems
        try:
            from .optimization.memory.object_pool import get_global_pool

            object_pool = get_global_pool()

            # Update pool sizes if available
            if hasattr(object_pool, "max_size"):
                object_pool.max_size = new_size
        except ImportError:
            pass

    async def _optimize_memory_usage(self, target_usage: float) -> None:
        """Optimize memory usage to target level."""
        if self._memory_optimizer:
            # Trigger memory optimization
            try:
                # Force garbage collection if memory pressure is high
                if target_usage < 0.5:  # Low target means high pressure
                    import gc

                    gc.collect()

                # Update memory optimizer settings
                self._memory_optimizer.track_object(
                    {"target_usage": target_usage}, "adaptive_memory_optimization"
                )
            except Exception:
                pass

    def add_adaptation_callback(self, callback: Callable[[AdaptationEvent], None]) -> None:
        """Add callback for adaptation events."""
        self._adaptation_callbacks.append(callback)

    def remove_adaptation_callback(self, callback: Callable[[AdaptationEvent], None]) -> None:
        """Remove adaptation callback."""
        if callback in self._adaptation_callbacks:
            self._adaptation_callbacks.remove(callback)

    def get_resource_limit(self, resource_type: ResourceType) -> Optional[ResourceLimit]:
        """Get current resource limit."""
        with self._lock:
            return self._resource_limits.get(resource_type)

    def set_resource_limit(
        self,
        resource_type: ResourceType,
        min_value: float,
        max_value: float,
        current_value: Optional[float] = None,
    ) -> None:
        """Set resource limit configuration."""
        with self._lock:
            if current_value is None:
                current_value = (min_value + max_value) / 2

            self._resource_limits[resource_type] = ResourceLimit(
                resource_type=resource_type,
                min_value=min_value,
                max_value=max_value,
                current_value=current_value,
            )

    def get_system_metrics(self) -> dict[ResourceType, ResourceMetrics]:
        """Get current system metrics."""
        return self._system_monitor.get_metrics()

    def get_adaptation_history(self, limit: int = 100) -> List[AdaptationEvent]:
        """Get recent adaptation history."""
        with self._lock:
            return list(self._adaptation_history)[-limit:]

    def get_stats(self) -> JSONObject:
        """Get resource manager statistics."""
        with self._lock:
            metrics = self.get_system_metrics()

            return {
                "adaptation_stats": self._stats.copy(),
                "current_metrics": {
                    resource_type.value: {
                        "current_usage": metric.current_usage,
                        "pressure_level": metric.pressure_level.value,
                        "trend": metric.get_trend(),
                    }
                    for resource_type, metric in metrics.items()
                },
                "resource_limits": {
                    resource_type.value: {
                        "min_value": limit.min_value,
                        "max_value": limit.max_value,
                        "current_value": limit.current_value,
                    }
                    for resource_type, limit in self._resource_limits.items()
                },
                "recent_adaptations": len(self._adaptation_history),
                "last_adaptation_time": self._last_adaptation_time,
            }

    def get_recommendations(self) -> List[JSONObject]:
        """Get resource optimization recommendations."""
        recommendations = []
        metrics = self.get_system_metrics()

        for resource_type, metric in metrics.items():
            if metric.pressure_level == PressureLevel.CRITICAL:
                recommendations.append(
                    {
                        "type": "critical_pressure",
                        "resource": resource_type.value,
                        "priority": "high",
                        "description": f"{resource_type.value} usage is critical ({metric.current_usage:.1%})",
                        "suggestion": "Immediate resource scaling or load reduction required",
                        "current_usage": metric.current_usage,
                        "trend": metric.get_trend(),
                    }
                )

            elif metric.pressure_level == PressureLevel.HIGH and metric.get_trend() == "increasing":
                recommendations.append(
                    {
                        "type": "increasing_pressure",
                        "resource": resource_type.value,
                        "priority": "medium",
                        "description": f"{resource_type.value} usage is high and increasing ({metric.current_usage:.1%})",
                        "suggestion": "Consider proactive resource scaling",
                        "current_usage": metric.current_usage,
                        "trend": metric.get_trend(),
                    }
                )

        # Check adaptation effectiveness
        recent_adaptations = self.get_adaptation_history(20)
        if recent_adaptations:
            failed_adaptations = [a for a in recent_adaptations if not a.success]
            if len(failed_adaptations) > len(recent_adaptations) * 0.3:
                recommendations.append(
                    {
                        "type": "adaptation_failure",
                        "priority": "medium",
                        "description": f"High adaptation failure rate ({len(failed_adaptations)}/{len(recent_adaptations)})",
                        "suggestion": "Review resource limits and adaptation strategies",
                    }
                )

        return recommendations


# Global adaptive resource manager instance
_global_adaptive_resource_manager: Optional[AdaptiveResourceManager] = None


def get_global_adaptive_resource_manager() -> AdaptiveResourceManager:
    """Get the global adaptive resource manager instance."""
    global _global_adaptive_resource_manager
    if _global_adaptive_resource_manager is None:
        _global_adaptive_resource_manager = AdaptiveResourceManager()
    return _global_adaptive_resource_manager


# Convenience functions
async def start_adaptive_resource_management() -> None:
    """Start adaptive resource management."""
    manager = get_global_adaptive_resource_manager()
    await manager.start()


async def stop_adaptive_resource_management() -> None:
    """Stop adaptive resource management."""
    manager = get_global_adaptive_resource_manager()
    await manager.stop()


def get_resource_metrics() -> dict[ResourceType, ResourceMetrics]:
    """Get current resource metrics."""
    manager = get_global_adaptive_resource_manager()
    return manager.get_system_metrics()


def get_resource_recommendations() -> List[JSONObject]:
    """Get resource optimization recommendations."""
    manager = get_global_adaptive_resource_manager()
    return manager.get_recommendations()


def set_resource_limit(
    resource_type: ResourceType,
    min_value: float,
    max_value: float,
    current_value: Optional[float] = None,
) -> None:
    """Set resource limit configuration."""
    manager = get_global_adaptive_resource_manager()
    manager.set_resource_limit(resource_type, min_value, max_value, current_value)
