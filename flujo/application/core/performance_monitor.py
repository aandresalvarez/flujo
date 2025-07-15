"""Performance monitoring for iterative step execution."""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ...infra import telemetry


@dataclass
class StepMetrics:
    """Metrics for a single step execution."""

    step_name: str
    execution_time: float
    cache_hit: bool = False
    retry_count: int = 0
    validation_time: float = 0.0
    memory_usage: Optional[float] = None
    cost_usd: float = 0.0
    token_count: int = 0


@dataclass
class PipelineMetrics:
    """Metrics for an entire pipeline execution."""

    total_steps: int = 0
    total_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    average_step_time: float = 0.0
    parallel_execution_time: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    step_metrics: List[StepMetrics] = field(default_factory=list)

    def add_step_metric(self, metric: StepMetrics) -> None:
        """Add a step metric to the pipeline metrics."""
        self.step_metrics.append(metric)
        self.total_steps += 1
        self.total_execution_time += metric.execution_time
        self.total_cost_usd += metric.cost_usd
        self.total_tokens += metric.token_count

        # Update cache hit rate
        cache_hits = sum(1 for m in self.step_metrics if m.cache_hit)
        self.cache_hit_rate = cache_hits / self.total_steps if self.total_steps > 0 else 0.0

        # Update average step time
        self.average_step_time = (
            self.total_execution_time / self.total_steps if self.total_steps > 0 else 0.0
        )


class PerformanceMonitor:
    """Monitor performance of iterative step execution."""

    def __init__(self) -> None:
        self.current_pipeline: Optional[PipelineMetrics] = None
        self.step_timings: Dict[str, List[float]] = defaultdict(list)
        self.cache_stats: Dict[str, int] = defaultdict(int)
        self.memory_usage: List[float] = []

    def start_pipeline(self) -> None:
        """Start monitoring a new pipeline execution."""
        self.current_pipeline = PipelineMetrics()

    def end_pipeline(self) -> Optional[PipelineMetrics]:
        """End monitoring and return pipeline metrics."""
        pipeline = self.current_pipeline
        self.current_pipeline = None
        return pipeline

    def record_step_start(self, step_name: str) -> float:
        """Record the start of a step execution."""
        return time.monotonic()

    def record_step_end(
        self,
        step_name: str,
        start_time: float,
        cache_hit: bool = False,
        retry_count: int = 0,
        validation_time: float = 0.0,
        cost_usd: float = 0.0,
        token_count: int = 0,
    ) -> StepMetrics:
        """Record the end of a step execution."""
        execution_time = time.monotonic() - start_time

        metric = StepMetrics(
            step_name=step_name,
            execution_time=execution_time,
            cache_hit=cache_hit,
            retry_count=retry_count,
            validation_time=validation_time,
            cost_usd=cost_usd,
            token_count=token_count,
        )

        if self.current_pipeline:
            self.current_pipeline.add_step_metric(metric)

        # Update global statistics
        self.step_timings[step_name].append(execution_time)
        if cache_hit:
            self.cache_stats[step_name] += 1

        # Log to telemetry
        with telemetry.logfire.span(f"step_execution:{step_name}") as span:
            span.set_attribute("execution_time", execution_time)
            span.set_attribute("cache_hit", cache_hit)
            span.set_attribute("retry_count", retry_count)
            span.set_attribute("validation_time", validation_time)
            span.set_attribute("cost_usd", cost_usd)
            span.set_attribute("token_count", token_count)

        return metric

    def get_step_statistics(self, step_name: str) -> Dict[str, Any]:
        """Get statistics for a specific step."""
        timings = self.step_timings[step_name]
        if not timings:
            return {}

        return {
            "count": len(timings),
            "mean_time": statistics.mean(timings),
            "median_time": statistics.median(timings),
            "min_time": min(timings),
            "max_time": max(timings),
            "std_dev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "cache_hits": self.cache_stats[step_name],
            "cache_hit_rate": self.cache_stats[step_name] / len(timings) if timings else 0.0,
        }

    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall execution statistics."""
        all_timings = []
        for timings in self.step_timings.values():
            all_timings.extend(timings)

        if not all_timings:
            return {}

        total_cache_hits = sum(self.cache_stats.values())
        total_executions = len(all_timings)

        return {
            "total_executions": total_executions,
            "total_cache_hits": total_cache_hits,
            "overall_cache_hit_rate": total_cache_hits / total_executions
            if total_executions > 0
            else 0.0,
            "mean_execution_time": statistics.mean(all_timings),
            "median_execution_time": statistics.median(all_timings),
            "min_execution_time": min(all_timings),
            "max_execution_time": max(all_timings),
            "std_dev_execution_time": statistics.stdev(all_timings)
            if len(all_timings) > 1
            else 0.0,
        }

    def reset(self) -> None:
        """Reset all monitoring data."""
        self.current_pipeline = None
        self.step_timings.clear()
        self.cache_stats.clear()
        self.memory_usage.clear()


# Global performance monitor instance
performance_monitor: PerformanceMonitor = PerformanceMonitor()
