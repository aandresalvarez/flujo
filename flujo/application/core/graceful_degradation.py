"""
Graceful degradation mechanisms for system resilience.

This module provides performance-based feature disabling, resource limit enforcement,
automatic recovery mechanisms, degradation level management, system health monitoring,
and automatic optimization adjustment for maintaining system stability under stress.
"""

import asyncio
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from asyncio import Task
from threading import RLock
from enum import Enum

from .optimized_telemetry import get_global_telemetry
from .adaptive_resource_manager import (
    ResourceType,
    PressureLevel,
    get_global_adaptive_resource_manager,
)
from .performance_monitor import get_global_performance_monitor


class DegradationLevel(Enum):
    """System degradation levels."""

    NORMAL = "normal"  # Full functionality
    MINOR = "minor"  # Non-essential features disabled
    MODERATE = "moderate"  # Significant feature reduction
    SEVERE = "severe"  # Core functionality only
    CRITICAL = "critical"  # Emergency mode


class FeatureCategory(Enum):
    """Categories of features for degradation."""

    ESSENTIAL = "essential"  # Core functionality
    IMPORTANT = "important"  # Important but not critical
    ENHANCEMENT = "enhancement"  # Performance enhancements
    OPTIONAL = "optional"  # Nice-to-have features
    EXPERIMENTAL = "experimental"  # Beta/experimental features


class DegradationTrigger(Enum):
    """Triggers for degradation."""

    CPU_PRESSURE = "cpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_PRESSURE = "disk_pressure"
    NETWORK_PRESSURE = "network_pressure"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MANUAL = "manual"


@dataclass
class Feature:
    """Feature definition for degradation management."""

    name: str
    category: FeatureCategory
    enabled: bool = True

    # Resource impact
    cpu_impact: float = 0.0  # CPU usage impact (0.0-1.0)
    memory_impact: float = 0.0  # Memory usage impact (0.0-1.0)
    io_impact: float = 0.0  # I/O impact (0.0-1.0)

    # Degradation settings
    disable_at_level: DegradationLevel = DegradationLevel.CRITICAL
    auto_recovery: bool = True
    recovery_delay: float = 30.0  # Seconds to wait before re-enabling

    # Callbacks
    disable_callback: Optional[Callable[[], None]] = None
    enable_callback: Optional[Callable[[], None]] = None

    # Statistics
    disable_count: int = 0
    last_disabled: Optional[float] = None
    total_disabled_time: float = 0.0

    def disable(self) -> bool:
        """Disable the feature."""
        if self.enabled:
            self.enabled = False
            self.disable_count += 1
            self.last_disabled = time.time()

            if self.disable_callback:
                try:
                    self.disable_callback()
                except Exception:
                    pass

            return True
        return False

    def enable(self) -> bool:
        """Enable the feature."""
        if not self.enabled:
            self.enabled = True

            if self.last_disabled:
                self.total_disabled_time += time.time() - self.last_disabled
                self.last_disabled = None

            if self.enable_callback:
                try:
                    self.enable_callback()
                except Exception:
                    pass

            return True
        return False

    @property
    def availability(self) -> float:
        """Calculate feature availability percentage."""
        total_time = time.time() - (self.last_disabled or time.time())
        if total_time <= 0:
            return 1.0

        disabled_time = self.total_disabled_time
        if self.last_disabled:
            disabled_time += time.time() - self.last_disabled

        return max(0.0, 1.0 - (disabled_time / total_time))


@dataclass
class DegradationEvent:
    """Record of a degradation event."""

    timestamp: float
    trigger: DegradationTrigger
    old_level: DegradationLevel
    new_level: DegradationLevel
    affected_features: List[str]
    resource_metrics: Dict[str, float]
    reason: str
    auto_triggered: bool = True

    @property
    def severity_change(self) -> int:
        """Calculate severity change (-4 to +4)."""
        levels = list(DegradationLevel)
        old_index = levels.index(self.old_level)
        new_index = levels.index(self.new_level)
        return new_index - old_index


@dataclass
class SystemHealthMetrics:
    """System health metrics for degradation decisions."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_usage: float = 0.0

    error_rate: float = 0.0
    average_response_time: float = 0.0
    active_connections: int = 0

    # Derived metrics
    overall_health_score: float = 1.0
    pressure_level: PressureLevel = PressureLevel.LOW

    def update_health_score(self) -> None:
        """Update overall health score."""
        # Weight different metrics
        cpu_score = 1.0 - min(self.cpu_usage, 1.0)
        memory_score = 1.0 - min(self.memory_usage, 1.0)
        disk_score = 1.0 - min(self.disk_usage, 1.0)
        network_score = 1.0 - min(self.network_usage, 1.0)
        error_score = 1.0 - min(self.error_rate, 1.0)

        # Response time score (assuming 1 second is baseline)
        response_score = max(0.0, 1.0 - (self.average_response_time / 1.0))

        # Weighted average
        self.overall_health_score = (
            cpu_score * 0.25
            + memory_score * 0.25
            + disk_score * 0.15
            + network_score * 0.15
            + error_score * 0.15
            + response_score * 0.05
        )

        # Determine pressure level
        if self.overall_health_score < 0.2:
            self.pressure_level = PressureLevel.CRITICAL
        elif self.overall_health_score < 0.4:
            self.pressure_level = PressureLevel.HIGH
        elif self.overall_health_score < 0.7:
            self.pressure_level = PressureLevel.MODERATE
        else:
            self.pressure_level = PressureLevel.LOW


class DegradationController:
    """
    Controller for graceful system degradation.

    Features:
    - Automatic feature disabling based on system pressure
    - Multiple degradation levels with configurable thresholds
    - Feature recovery with hysteresis
    - Resource impact calculation
    - Health monitoring integration
    - Manual override capabilities
    """

    def __init__(
        self,
        monitoring_interval: float = 2.0,
        recovery_check_interval: float = 10.0,
        enable_telemetry: bool = True,
    ):
        self.monitoring_interval = monitoring_interval
        self.recovery_check_interval = recovery_check_interval
        self.enable_telemetry = enable_telemetry

        # Core components
        self._telemetry = get_global_telemetry() if enable_telemetry else None
        self._resource_manager = get_global_adaptive_resource_manager()
        self._performance_monitor = get_global_performance_monitor()

        # Degradation state
        self._current_level = DegradationLevel.NORMAL
        self._features: Dict[str, Feature] = {}
        self._degradation_history: deque[DegradationEvent] = deque(maxlen=1000)

        # Performance tracking
        self._degradation_events: deque[DegradationEvent] = deque(maxlen=100)
        self._recovery_events: deque[DegradationEvent] = deque(maxlen=100)

        # Thresholds for degradation levels
        self._degradation_thresholds = {
            DegradationLevel.MINOR: 0.8,  # 80% health score
            DegradationLevel.MODERATE: 0.6,  # 60% health score
            DegradationLevel.SEVERE: 0.4,  # 40% health score
            DegradationLevel.CRITICAL: 0.2,  # 20% health score
        }

        # Recovery thresholds (with hysteresis)
        self._recovery_thresholds = {
            DegradationLevel.NORMAL: 0.85,  # 85% health score
            DegradationLevel.MINOR: 0.75,  # 75% health score
            DegradationLevel.MODERATE: 0.55,  # 55% health score
            DegradationLevel.SEVERE: 0.35,  # 35% health score
        }

        # Monitoring tasks
        self._monitoring_task: Optional[Task[Any]] = None
        self._recovery_task: Optional[Task[Any]] = None
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._degradation_callbacks: List[Callable[[DegradationEvent], None]] = []

        # Statistics
        self._stats: Dict[str, Any] = {
            "total_degradations": 0,
            "total_recoveries": 0,
            "features_disabled": defaultdict(int),
            "degradation_triggers": defaultdict(int),
            "time_in_degradation": defaultdict(float),
        }

        # Thread safety
        self._lock = RLock()

        # Initialize default features
        self._initialize_default_features()

    def _initialize_default_features(self) -> None:
        """Initialize default system features."""
        default_features = [
            Feature(
                name="object_pooling",
                category=FeatureCategory.ENHANCEMENT,
                cpu_impact=0.05,
                memory_impact=0.1,
                disable_at_level=DegradationLevel.SEVERE,
            ),
            Feature(
                name="context_optimization",
                category=FeatureCategory.ENHANCEMENT,
                cpu_impact=0.03,
                memory_impact=0.05,
                disable_at_level=DegradationLevel.MODERATE,
            ),
            Feature(
                name="step_caching",
                category=FeatureCategory.IMPORTANT,
                cpu_impact=0.02,
                memory_impact=0.15,
                disable_at_level=DegradationLevel.SEVERE,
            ),
            Feature(
                name="performance_monitoring",
                category=FeatureCategory.OPTIONAL,
                cpu_impact=0.01,
                memory_impact=0.02,
                disable_at_level=DegradationLevel.MINOR,
            ),
            Feature(
                name="detailed_telemetry",
                category=FeatureCategory.OPTIONAL,
                cpu_impact=0.02,
                memory_impact=0.03,
                disable_at_level=DegradationLevel.MINOR,
            ),
            Feature(
                name="algorithm_optimizations",
                category=FeatureCategory.ENHANCEMENT,
                cpu_impact=0.01,
                memory_impact=0.01,
                disable_at_level=DegradationLevel.MODERATE,
            ),
            Feature(
                name="concurrency_optimization",
                category=FeatureCategory.IMPORTANT,
                cpu_impact=0.05,
                memory_impact=0.02,
                disable_at_level=DegradationLevel.SEVERE,
            ),
            Feature(
                name="error_recovery",
                category=FeatureCategory.ESSENTIAL,
                cpu_impact=0.01,
                memory_impact=0.01,
                disable_at_level=DegradationLevel.CRITICAL,
            ),
        ]

        for feature in default_features:
            self._features[feature.name] = feature

    async def start(self) -> None:
        """Start degradation monitoring."""
        self._shutdown_event.clear()

        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        if self._recovery_task is None:
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop(self) -> None:
        """Stop degradation monitoring."""
        self._shutdown_event.set()

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
            self._recovery_task = None

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_system_health()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _recovery_loop(self) -> None:
        """Recovery monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_recovery_conditions()
                await asyncio.sleep(self.recovery_check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _check_system_health(self) -> None:
        """Check system health and trigger degradation if needed."""
        # Collect system metrics
        health_metrics = await self._collect_health_metrics()

        # Determine required degradation level
        required_level = self._calculate_required_degradation_level(health_metrics)

        # Apply degradation if needed
        if required_level != self._current_level:
            await self._apply_degradation(required_level, health_metrics)

    async def _collect_health_metrics(self) -> SystemHealthMetrics:
        """Collect comprehensive system health metrics."""
        metrics = SystemHealthMetrics()

        try:
            # Get resource metrics
            resource_metrics = self._resource_manager.get_system_metrics()

            if ResourceType.CPU in resource_metrics:
                metrics.cpu_usage = resource_metrics[ResourceType.CPU].current_usage

            if ResourceType.MEMORY in resource_metrics:
                metrics.memory_usage = resource_metrics[ResourceType.MEMORY].current_usage

            if ResourceType.DISK_IO in resource_metrics:
                metrics.disk_usage = resource_metrics[ResourceType.DISK_IO].current_usage

            if ResourceType.NETWORK_IO in resource_metrics:
                metrics.network_usage = resource_metrics[ResourceType.NETWORK_IO].current_usage

            # Get performance metrics
            if self._performance_monitor:
                perf_summary = self._performance_monitor.get_performance_summary()

                # Extract error rate and response time
                if "application_metrics" in perf_summary:
                    app_metrics = perf_summary["application_metrics"]

                    # Calculate average response time from executor metrics
                    if "executor.execution_time_ms" in app_metrics:
                        exec_metrics = app_metrics["executor.execution_time_ms"]
                        metrics.average_response_time = exec_metrics.get("mean", 0.0) / 1000.0

                    # Calculate error rate
                    if (
                        "executor.errors" in app_metrics
                        and "executor.executions_total" in app_metrics
                    ):
                        errors = app_metrics["executor.errors"].get("count", 0)
                        total = app_metrics["executor.executions_total"].get("count", 1)
                        metrics.error_rate = errors / max(total, 1)

            # Update overall health score
            metrics.update_health_score()

        except Exception:
            # Use default metrics if collection fails
            pass

        return metrics

    def _calculate_required_degradation_level(
        self, metrics: SystemHealthMetrics
    ) -> DegradationLevel:
        """Calculate required degradation level based on health metrics."""
        health_score = metrics.overall_health_score

        # Check degradation thresholds (from most severe to least)
        for level in [
            DegradationLevel.CRITICAL,
            DegradationLevel.SEVERE,
            DegradationLevel.MODERATE,
            DegradationLevel.MINOR,
        ]:
            if health_score <= self._degradation_thresholds[level]:
                return level

        return DegradationLevel.NORMAL

    async def _apply_degradation(
        self, target_level: DegradationLevel, metrics: SystemHealthMetrics
    ) -> None:
        """Apply system degradation to target level."""
        old_level = self._current_level

        # Determine trigger
        trigger = self._determine_degradation_trigger(metrics)

        # Get features to disable/enable
        features_to_disable = []
        features_to_enable = []

        if target_level.value > old_level.value:  # Increasing degradation
            # Disable features that should be disabled at this level
            for feature in self._features.values():
                if feature.enabled and feature.disable_at_level.value <= target_level.value:
                    features_to_disable.append(feature.name)

        else:  # Decreasing degradation (recovery)
            # Enable features that can be enabled at this level
            for feature in self._features.values():
                if (
                    not feature.enabled
                    and feature.disable_at_level.value > target_level.value
                    and feature.auto_recovery
                ):
                    features_to_enable.append(feature.name)

        # Apply feature changes
        for feature_name in features_to_disable:
            await self._disable_feature(feature_name)

        for feature_name in features_to_enable:
            await self._enable_feature(feature_name)

        # Update current level
        self._current_level = target_level

        # Create degradation event
        event = DegradationEvent(
            timestamp=time.time(),
            trigger=trigger,
            old_level=old_level,
            new_level=target_level,
            affected_features=features_to_disable + features_to_enable,
            resource_metrics={
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "health_score": metrics.overall_health_score,
            },
            reason=f"Health score: {metrics.overall_health_score:.2f}, Pressure: {metrics.pressure_level.value}",
        )

        # Record event
        with self._lock:
            self._degradation_history.append(event)

            if target_level.value > old_level.value:
                self._stats["total_degradations"] += 1
            else:
                self._stats["total_recoveries"] += 1

            self._stats["degradation_triggers"][trigger.value] += 1

        # Notify callbacks
        for callback in self._degradation_callbacks:
            try:
                callback(event)
            except Exception:
                continue

        # Record telemetry
        if self._telemetry:
            self._telemetry.record_metric(
                "degradation.level_change",
                list(DegradationLevel).index(target_level),
                tags={
                    "old_level": old_level.value,
                    "new_level": target_level.value,
                    "trigger": trigger.value,
                },
            )

            self._telemetry.record_metric(
                "degradation.health_score",
                metrics.overall_health_score,
                tags={"degradation_level": target_level.value},
            )

    def _determine_degradation_trigger(self, metrics: SystemHealthMetrics) -> DegradationTrigger:
        """Determine the primary trigger for degradation."""
        # Find the resource with highest usage
        resource_usage = {
            "cpu": metrics.cpu_usage,
            "memory": metrics.memory_usage,
            "disk": metrics.disk_usage,
            "network": metrics.network_usage,
        }

        max_resource = max(resource_usage.items(), key=lambda x: x[1])

        if max_resource[1] > 0.8:
            if max_resource[0] == "cpu":
                return DegradationTrigger.CPU_PRESSURE
            elif max_resource[0] == "memory":
                return DegradationTrigger.MEMORY_PRESSURE
            elif max_resource[0] == "disk":
                return DegradationTrigger.DISK_PRESSURE
            elif max_resource[0] == "network":
                return DegradationTrigger.NETWORK_PRESSURE

        if metrics.error_rate > 0.1:
            return DegradationTrigger.ERROR_RATE

        if metrics.average_response_time > 2.0:
            return DegradationTrigger.RESPONSE_TIME

        return DegradationTrigger.RESOURCE_EXHAUSTION

    async def _disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature."""
        with self._lock:
            if feature_name in self._features:
                feature = self._features[feature_name]
                if feature.disable():
                    self._stats["features_disabled"][feature_name] += 1

                    if self._telemetry:
                        self._telemetry.increment_counter(
                            "degradation.feature_disabled",
                            tags={"feature": feature_name, "category": feature.category.value},
                        )

                    return True
        return False

    async def _enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature."""
        with self._lock:
            if feature_name in self._features:
                feature = self._features[feature_name]
                if feature.enable():
                    if self._telemetry:
                        self._telemetry.increment_counter(
                            "degradation.feature_enabled",
                            tags={"feature": feature_name, "category": feature.category.value},
                        )

                    return True
        return False

    async def _check_recovery_conditions(self) -> None:
        """Check if system can recover from current degradation level."""
        if self._current_level == DegradationLevel.NORMAL:
            return

        # Collect current health metrics
        metrics = await self._collect_health_metrics()

        # Check if we can recover to a better level
        recovery_threshold = self._recovery_thresholds.get(self._current_level, 0.85)

        if metrics.overall_health_score >= recovery_threshold:
            # Determine target recovery level
            target_level = DegradationLevel.NORMAL

            for level in [
                DegradationLevel.MINOR,
                DegradationLevel.MODERATE,
                DegradationLevel.SEVERE,
            ]:
                if metrics.overall_health_score < self._recovery_thresholds[level]:
                    target_level = level
                    break

            if target_level.value < self._current_level.value:
                await self._apply_degradation(target_level, metrics)

    def register_feature(
        self,
        name: str,
        category: FeatureCategory,
        cpu_impact: float = 0.0,
        memory_impact: float = 0.0,
        io_impact: float = 0.0,
        disable_at_level: DegradationLevel = DegradationLevel.CRITICAL,
        disable_callback: Optional[Callable[[], None]] = None,
        enable_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a feature for degradation management."""
        feature = Feature(
            name=name,
            category=category,
            cpu_impact=cpu_impact,
            memory_impact=memory_impact,
            io_impact=io_impact,
            disable_at_level=disable_at_level,
            disable_callback=disable_callback,
            enable_callback=enable_callback,
        )

        with self._lock:
            self._features[name] = feature

    def unregister_feature(self, name: str) -> None:
        """Unregister a feature."""
        with self._lock:
            if name in self._features:
                del self._features[name]

    def is_feature_enabled(self, name: str) -> bool:
        """Check if a feature is currently enabled."""
        with self._lock:
            if name in self._features:
                return self._features[name].enabled
            return True  # Unknown features are considered enabled

    def force_degradation(self, level: DegradationLevel, reason: str = "Manual override") -> None:
        """Force system to specific degradation level."""
        asyncio.create_task(self._force_degradation_async(level, reason))

    async def _force_degradation_async(self, level: DegradationLevel, reason: str) -> None:
        """Async implementation of force degradation."""
        metrics = await self._collect_health_metrics()

        # Create manual event
        # event = DegradationEvent(  # Unused variable
        #     timestamp=time.time(),
        #     trigger=DegradationTrigger.MANUAL,
        #     old_level=self._current_level,
        #     new_level=level,
        #     affected_features=[],
        #     resource_metrics={
        #         "cpu_usage": metrics.cpu_usage,
        #         "memory_usage": metrics.memory_usage,
        #         "health_score": metrics.overall_health_score,
        #     },
        #     reason=reason,
        #     auto_triggered=False,
        # )

        await self._apply_degradation(level, metrics)

    def add_degradation_callback(self, callback: Callable[[DegradationEvent], None]) -> None:
        """Add callback for degradation events."""
        self._degradation_callbacks.append(callback)

    def remove_degradation_callback(self, callback: Callable[[DegradationEvent], None]) -> None:
        """Remove degradation callback."""
        if callback in self._degradation_callbacks:
            self._degradation_callbacks.remove(callback)

    def get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_level

    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all features."""
        with self._lock:
            return {
                name: {
                    "enabled": feature.enabled,
                    "category": feature.category.value,
                    "disable_at_level": feature.disable_at_level.value,
                    "disable_count": feature.disable_count,
                    "availability": feature.availability,
                }
                for name, feature in self._features.items()
            }

    def get_degradation_history(self, limit: int = 100) -> List[DegradationEvent]:
        """Get recent degradation history."""
        with self._lock:
            return list(self._degradation_history)[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get degradation controller statistics."""
        with self._lock:
            return {
                "current_level": self._current_level.value,
                "degradation_stats": self._stats.copy(),
                "feature_status": self.get_feature_status(),
                "recent_events": len(self._degradation_history),
                "thresholds": {
                    "degradation": {
                        level.value: threshold
                        for level, threshold in self._degradation_thresholds.items()
                    },
                    "recovery": {
                        level.value: threshold
                        for level, threshold in self._recovery_thresholds.items()
                    },
                },
            }


# Global degradation controller instance
_global_degradation_controller: Optional[DegradationController] = None


def get_global_degradation_controller() -> DegradationController:
    """Get the global degradation controller instance."""
    global _global_degradation_controller
    if _global_degradation_controller is None:
        _global_degradation_controller = DegradationController()
    return _global_degradation_controller


# Convenience functions
async def start_graceful_degradation() -> None:
    """Start graceful degradation monitoring."""
    controller = get_global_degradation_controller()
    await controller.start()


async def stop_graceful_degradation() -> None:
    """Stop graceful degradation monitoring."""
    controller = get_global_degradation_controller()
    await controller.stop()


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is currently enabled."""
    controller = get_global_degradation_controller()
    return controller.is_feature_enabled(feature_name)


def register_feature(
    name: str,
    category: FeatureCategory,
    cpu_impact: float = 0.0,
    memory_impact: float = 0.0,
    disable_at_level: DegradationLevel = DegradationLevel.CRITICAL,
    disable_callback: Optional[Callable[[], None]] = None,
    enable_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Register a feature for degradation management."""
    controller = get_global_degradation_controller()
    controller.register_feature(
        name,
        category,
        cpu_impact,
        memory_impact,
        0.0,
        disable_at_level,
        disable_callback,
        enable_callback,
    )


def get_current_degradation_level() -> DegradationLevel:
    """Get current system degradation level."""
    controller = get_global_degradation_controller()
    return controller.get_current_level()


def force_degradation(level: DegradationLevel, reason: str = "Manual override") -> None:
    """Force system to specific degradation level."""
    controller = get_global_degradation_controller()
    controller.force_degradation(level, reason)


def get_degradation_stats() -> Dict[str, Any]:
    """Get degradation system statistics."""
    controller = get_global_degradation_controller()
    return controller.get_stats()
