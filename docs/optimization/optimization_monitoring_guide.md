# Optimization Monitoring and Metrics Guide

## Overview

This guide covers how to monitor and analyze the performance of ExecutorCore optimizations. It includes metrics collection, performance analysis, alerting, and visualization techniques.

## Table of Contents

1. [Available Metrics](#available-metrics)
2. [Performance Monitoring Setup](#performance-monitoring-setup)
3. [Real-Time Monitoring](#real-time-monitoring)
4. [Performance Analysis](#performance-analysis)
5. [Alerting and Thresholds](#alerting-and-thresholds)
6. [Visualization and Reporting](#visualization-and-reporting)
7. [Best Practices](#best-practices)

## Available Metrics

### Memory Metrics

#### Object Pool Metrics
```python
# Get object pool statistics
stats = executor.get_optimization_stats()
memory_stats = stats.get('memory', {})

print(f"Object pool utilization: {memory_stats.get('object_pool_utilization', 'N/A')}%")
print(f"Objects in pool: {memory_stats.get('objects_in_pool', 'N/A')}")
print(f"Pool hit rate: {memory_stats.get('pool_hit_rate', 'N/A')}%")
print(f"Pool miss rate: {memory_stats.get('pool_miss_rate', 'N/A')}%")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `object_pool_utilization` | Percentage of pool capacity used | % | >80% (high utilization) |
| `objects_in_pool` | Number of objects currently in pool | count | Varies by type |
| `pool_hit_rate` | Percentage of object requests served from pool | % | <50% (low efficiency) |
| `pool_miss_rate` | Percentage of object requests requiring allocation | % | >50% (high allocation) |

#### Memory Usage Metrics
```python
# Get memory usage statistics
memory_stats = stats.get('memory', {})

print(f"Memory usage: {memory_stats.get('usage_mb', 'N/A')}MB")
print(f"Memory pressure level: {memory_stats.get('pressure_level', 'N/A')}")
print(f"Memory allocation rate: {memory_stats.get('allocation_rate_mb_s', 'N/A')}MB/s")
print(f"Memory cleanup events: {memory_stats.get('cleanup_events', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `usage_mb` | Current memory usage | MB | >500MB (high usage) |
| `pressure_level` | Memory pressure indicator | level | >0.8 (high pressure) |
| `allocation_rate_mb_s` | Memory allocation rate | MB/s | >10MB/s (high allocation) |
| `cleanup_events` | Number of cleanup events | count | Varies |

### Performance Metrics

#### Execution Metrics
```python
# Get execution performance statistics
execution_stats = stats.get('execution', {})

print(f"Average execution time: {execution_stats.get('avg_time_ms', 'N/A')}ms")
print(f"Min execution time: {execution_stats.get('min_time_ms', 'N/A')}ms")
print(f"Max execution time: {execution_stats.get('max_time_ms', 'N/A')}ms")
print(f"Total executions: {execution_stats.get('total_executions', 'N/A')}")
print(f"Successful executions: {execution_stats.get('successful_executions', 'N/A')}")
print(f"Failed executions: {execution_stats.get('failed_executions', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `avg_time_ms` | Average execution time | ms | >1000ms (slow) |
| `min_time_ms` | Minimum execution time | ms | Baseline |
| `max_time_ms` | Maximum execution time | ms | >5000ms (very slow) |
| `total_executions` | Total number of executions | count | Varies |
| `successful_executions` | Number of successful executions | count | Varies |
| `failed_executions` | Number of failed executions | count | >10% (high failure rate) |

#### Cache Metrics
```python
# Get cache performance statistics
cache_stats = stats.get('cache', {})

print(f"Cache hit rate: {cache_stats.get('hit_rate', 'N/A')}%")
print(f"Cache size: {cache_stats.get('size', 'N/A')}")
print(f"Cache utilization: {cache_stats.get('utilization', 'N/A')}%")
print(f"Cache evictions: {cache_stats.get('evictions', 'N/A')}")
print(f"Cache misses: {cache_stats.get('misses', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `hit_rate` | Cache hit rate percentage | % | <30% (low efficiency) |
| `size` | Current cache size | count | >80% capacity |
| `utilization` | Cache utilization percentage | % | >90% (high utilization) |
| `evictions` | Number of cache evictions | count | High (frequent evictions) |
| `misses` | Number of cache misses | count | High (frequent misses) |

### Concurrency Metrics

#### Concurrency Performance
```python
# Get concurrency statistics
concurrency_stats = stats.get('concurrency', {})

print(f"Current concurrency level: {concurrency_stats.get('current_level', 'N/A')}")
print(f"Max concurrency level: {concurrency_stats.get('max_level', 'N/A')}")
print(f"Average wait time: {concurrency_stats.get('avg_wait_time_ms', 'N/A')}ms")
print(f"Contention rate: {concurrency_stats.get('contention_rate', 'N/A')}%")
print(f"Work queue size: {concurrency_stats.get('queue_size', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `current_level` | Current concurrency level | count | >80% of max |
| `max_level` | Maximum concurrency level | count | Varies |
| `avg_wait_time_ms` | Average wait time for execution | ms | >100ms (high contention) |
| `contention_rate` | Rate of concurrency contention | % | >20% (high contention) |
| `queue_size` | Current work queue size | count | >50 (backlog) |

### Error Handling Metrics

#### Error Statistics
```python
# Get error handling statistics
error_stats = stats.get('error_handling', {})

print(f"Error rate: {error_stats.get('error_rate', 'N/A')}%")
print(f"Recovery success rate: {error_stats.get('recovery_success_rate', 'N/A')}%")
print(f"Cached error patterns: {error_stats.get('cached_patterns', 'N/A')}")
print(f"Circuit breaker state: {error_stats.get('circuit_breaker_state', 'N/A')}")
print(f"Failure count: {error_stats.get('failure_count', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `error_rate` | Percentage of failed executions | % | >10% (high error rate) |
| `recovery_success_rate` | Percentage of successful recoveries | % | <50% (poor recovery) |
| `cached_patterns` | Number of cached error patterns | count | Varies |
| `circuit_breaker_state` | Current circuit breaker state | state | 'OPEN' (circuit open) |
| `failure_count` | Number of consecutive failures | count | >5 (high failure count) |

### Telemetry Metrics

#### Telemetry Performance
```python
# Get telemetry statistics
telemetry_stats = stats.get('telemetry', {})

print(f"Telemetry batch size: {telemetry_stats.get('batch_size', 'N/A')}")
print(f"Telemetry flush rate: {telemetry_stats.get('flush_rate', 'N/A')}/s")
print(f"Telemetry overhead: {telemetry_stats.get('overhead_ms', 'N/A')}ms")
print(f"Metrics collected: {telemetry_stats.get('metrics_collected', 'N/A')}")
```

| Metric | Description | Unit | Threshold |
|--------|-------------|------|-----------|
| `batch_size` | Current telemetry batch size | count | Varies |
| `flush_rate` | Telemetry flush rate | flushes/s | Varies |
| `overhead_ms` | Telemetry overhead in milliseconds | ms | >10ms (high overhead) |
| `metrics_collected` | Number of metrics collected | count | Varies |

## Performance Monitoring Setup

### Basic Monitoring Setup

```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
import asyncio
import time

# Configure monitoring
config = OptimizationConfig(
    enable_performance_monitoring=True,
    enable_optimized_telemetry=True,
    telemetry_batch_size=10,
    telemetry_flush_interval_seconds=30.0,
)

# Create executor with monitoring
executor = ExecutorCore(optimization_config=config)

# Basic monitoring function
async def basic_monitor(executor):
    while True:
        stats = executor.get_optimization_stats()

        # Print key metrics
        print(f"Memory: {stats.get('memory', {}).get('usage_mb', 'N/A')}MB")
        print(f"Execution time: {stats.get('execution', {}).get('avg_time_ms', 'N/A')}ms")
        print(f"Cache hit rate: {stats.get('cache', {}).get('hit_rate', 'N/A')}%")
        print(f"Error rate: {stats.get('error_handling', {}).get('error_rate', 'N/A')}%")
        print("-" * 50)

        await asyncio.sleep(10)
```

### Advanced Monitoring Setup

```python
import json
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, executor, alert_thresholds=None):
        self.executor = executor
        self.alert_thresholds = alert_thresholds or {
            'memory_usage_mb': 500,
            'execution_time_ms': 1000,
            'error_rate': 0.1,
            'cache_hit_rate': 0.3,
        }
        self.metrics_history = []

    async def monitor(self, interval=10):
        """Monitor performance metrics continuously."""
        while True:
            try:
                # Collect metrics
                stats = self.executor.get_optimization_stats()
                timestamp = datetime.now(timezone.utc).isoformat()

                # Store metrics
                metrics = {
                    'timestamp': timestamp,
                    'memory': stats.get('memory', {}),
                    'execution': stats.get('execution', {}),
                    'cache': stats.get('cache', {}),
                    'error_handling': stats.get('error_handling', {}),
                    'concurrency': stats.get('concurrency', {}),
                    'telemetry': stats.get('telemetry', {}),
                }

                self.metrics_history.append(metrics)

                # Check for alerts
                await self._check_alerts(metrics)

                # Log metrics
                logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")

            except Exception as e:
                logger.error(f"Error in monitoring: {e}")

            await asyncio.sleep(interval)

    async def _check_alerts(self, metrics):
        """Check for performance alerts."""
        alerts = []

        # Memory alerts
        memory_usage = metrics['memory'].get('usage_mb', 0)
        if memory_usage > self.alert_thresholds['memory_usage_mb']:
            alerts.append(f"High memory usage: {memory_usage}MB")

        # Execution time alerts
        avg_time = metrics['execution'].get('avg_time_ms', 0)
        if avg_time > self.alert_thresholds['execution_time_ms']:
            alerts.append(f"Slow execution: {avg_time}ms")

        # Error rate alerts
        error_rate = metrics['error_handling'].get('error_rate', 0)
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {error_rate:.2%}")

        # Cache hit rate alerts
        hit_rate = metrics['cache'].get('hit_rate', 1.0)
        if hit_rate < self.alert_thresholds['cache_hit_rate']:
            alerts.append(f"Low cache hit rate: {hit_rate:.2%}")

        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")

    def get_metrics_summary(self):
        """Get a summary of recent metrics."""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements

        summary = {
            'memory_avg': sum(m['memory'].get('usage_mb', 0) for m in recent_metrics) / len(recent_metrics),
            'execution_avg': sum(m['execution'].get('avg_time_ms', 0) for m in recent_metrics) / len(recent_metrics),
            'error_rate_avg': sum(m['error_handling'].get('error_rate', 0) for m in recent_metrics) / len(recent_metrics),
            'cache_hit_rate_avg': sum(m['cache'].get('hit_rate', 1.0) for m in recent_metrics) / len(recent_metrics),
        }

        return summary
```

## Real-Time Monitoring

### Live Dashboard

```python
import asyncio
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live

class LiveDashboard:
    def __init__(self, executor):
        self.executor = executor
        self.console = Console()

    def generate_table(self, stats):
        """Generate a rich table with current metrics."""
        table = Table(title="ExecutorCore Optimization Metrics")

        # Add columns
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        # Memory metrics
        memory_stats = stats.get('memory', {})
        memory_usage = memory_stats.get('usage_mb', 'N/A')
        table.add_row("Memory Usage", f"{memory_usage}MB",
                     "⚠️" if memory_usage > 500 else "✅")

        # Execution metrics
        execution_stats = stats.get('execution', {})
        avg_time = execution_stats.get('avg_time_ms', 'N/A')
        table.add_row("Avg Execution Time", f"{avg_time}ms",
                     "⚠️" if avg_time > 1000 else "✅")

        # Cache metrics
        cache_stats = stats.get('cache', {})
        hit_rate = cache_stats.get('hit_rate', 'N/A')
        table.add_row("Cache Hit Rate", f"{hit_rate}%",
                     "⚠️" if hit_rate < 30 else "✅")

        # Error metrics
        error_stats = stats.get('error_handling', {})
        error_rate = error_stats.get('error_rate', 'N/A')
        table.add_row("Error Rate", f"{error_rate}%",
                     "⚠️" if error_rate > 10 else "✅")

        # Concurrency metrics
        concurrency_stats = stats.get('concurrency', {})
        current_level = concurrency_stats.get('current_level', 'N/A')
        table.add_row("Concurrency Level", f"{current_level}",
                     "✅")

        return table

    async def run_dashboard(self):
        """Run the live dashboard."""
        with Live(self.generate_table({}), refresh_per_second=1) as live:
            while True:
                stats = self.executor.get_optimization_stats()
                live.update(self.generate_table(stats))
                await asyncio.sleep(1)
```

### Performance Tracking

```python
import csv
from datetime import datetime, timezone

class PerformanceTracker:
    def __init__(self, filename="performance_metrics.csv"):
        self.filename = filename
        self.fieldnames = [
            'timestamp', 'memory_usage_mb', 'execution_time_ms',
            'cache_hit_rate', 'error_rate', 'concurrency_level'
        ]

        # Create CSV file with headers
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def record_metrics(self, stats):
        """Record metrics to CSV file."""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'memory_usage_mb': stats.get('memory', {}).get('usage_mb', 0),
            'execution_time_ms': stats.get('execution', {}).get('avg_time_ms', 0),
            'cache_hit_rate': stats.get('cache', {}).get('hit_rate', 0),
            'error_rate': stats.get('error_handling', {}).get('error_rate', 0),
            'concurrency_level': stats.get('concurrency', {}).get('current_level', 0),
        }

        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(metrics)

    async def track_performance(self, executor, interval=10):
        """Track performance metrics continuously."""
        while True:
            stats = executor.get_optimization_stats()
            self.record_metrics(stats)
            await asyncio.sleep(interval)
```

## Performance Analysis

### Trend Analysis

```python
import numpy as np
from collections import deque

class TrendAnalyzer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)

    def add_metrics(self, metrics):
        """Add metrics to the analysis window."""
        self.metrics_history.append(metrics)

    def analyze_trends(self):
        """Analyze performance trends."""
        if len(self.metrics_history) < 10:
            return "Insufficient data for trend analysis"

        # Extract time series data
        memory_usage = [m['memory'].get('usage_mb', 0) for m in self.metrics_history]
        execution_time = [m['execution'].get('avg_time_ms', 0) for m in self.metrics_history]
        error_rate = [m['error_handling'].get('error_rate', 0) for m in self.metrics_history]

        # Calculate trends
        memory_trend = self._calculate_trend(memory_usage)
        execution_trend = self._calculate_trend(execution_time)
        error_trend = self._calculate_trend(error_rate)

        return {
            'memory_trend': memory_trend,
            'execution_trend': execution_trend,
            'error_trend': error_trend,
            'recommendations': self._generate_recommendations(memory_trend, execution_trend, error_trend)
        }

    def _calculate_trend(self, data):
        """Calculate trend direction and magnitude."""
        if len(data) < 2:
            return "insufficient_data"

        # Calculate linear regression
        x = np.arange(len(data))
        y = np.array(data)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _generate_recommendations(self, memory_trend, execution_trend, error_trend):
        """Generate recommendations based on trends."""
        recommendations = []

        if memory_trend == "increasing":
            recommendations.append("Memory usage is increasing - consider reducing object pool size")

        if execution_trend == "increasing":
            recommendations.append("Execution time is increasing - check for performance bottlenecks")

        if error_trend == "increasing":
            recommendations.append("Error rate is increasing - investigate error sources")

        return recommendations
```

### Performance Regression Detection

```python
class RegressionDetector:
    def __init__(self, baseline_metrics, threshold=0.2):
        self.baseline_metrics = baseline_metrics
        self.threshold = threshold

    def detect_regression(self, current_metrics):
        """Detect performance regression compared to baseline."""
        regressions = []

        # Check memory usage
        baseline_memory = self.baseline_metrics.get('memory_usage_mb', 0)
        current_memory = current_metrics.get('memory_usage_mb', 0)
        if current_memory > baseline_memory * (1 + self.threshold):
            regressions.append(f"Memory usage regression: {current_memory}MB vs {baseline_memory}MB")

        # Check execution time
        baseline_time = self.baseline_metrics.get('execution_time_ms', 0)
        current_time = current_metrics.get('execution_time_ms', 0)
        if current_time > baseline_time * (1 + self.threshold):
            regressions.append(f"Execution time regression: {current_time}ms vs {baseline_time}ms")

        # Check error rate
        baseline_error = self.baseline_metrics.get('error_rate', 0)
        current_error = current_metrics.get('error_rate', 0)
        if current_error > baseline_error * (1 + self.threshold):
            regressions.append(f"Error rate regression: {current_error:.2%} vs {baseline_error:.2%}")

        return regressions
```

## Alerting and Thresholds

### Alert Configuration

```python
class AlertManager:
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'critical': {
                'memory_usage_mb': 800,
                'execution_time_ms': 5000,
                'error_rate': 0.3,
                'cache_hit_rate': 0.1,
            },
            'warning': {
                'memory_usage_mb': 500,
                'execution_time_ms': 1000,
                'error_rate': 0.1,
                'cache_hit_rate': 0.3,
            }
        }

    def check_alerts(self, metrics):
        """Check for performance alerts."""
        alerts = []

        # Check critical thresholds
        for metric, threshold in self.thresholds['critical'].items():
            value = self._get_metric_value(metrics, metric)
            if value is not None and value > threshold:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'message': f"Critical {metric}: {value} (threshold: {threshold})"
                })

        # Check warning thresholds
        for metric, threshold in self.thresholds['warning'].items():
            value = self._get_metric_value(metrics, metric)
            if value is not None and value > threshold:
                alerts.append({
                    'level': 'WARNING',
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'message': f"Warning {metric}: {value} (threshold: {threshold})"
                })

        return alerts

    def _get_metric_value(self, metrics, metric_name):
        """Extract metric value from metrics dictionary."""
        metric_mapping = {
            'memory_usage_mb': ('memory', 'usage_mb'),
            'execution_time_ms': ('execution', 'avg_time_ms'),
            'error_rate': ('error_handling', 'error_rate'),
            'cache_hit_rate': ('cache', 'hit_rate'),
        }

        if metric_name in metric_mapping:
            category, field = metric_mapping[metric_name]
            return metrics.get(category, {}).get(field)

        return None
```

### Alert Handlers

```python
import smtplib
from email.mime.text import MIMEText

class AlertHandler:
    def __init__(self, email_config=None):
        self.email_config = email_config

    async def handle_alerts(self, alerts):
        """Handle performance alerts."""
        for alert in alerts:
            # Log alert
            logger.warning(f"ALERT: {alert['message']}")

            # Send email if configured
            if self.email_config and alert['level'] == 'CRITICAL':
                await self._send_email_alert(alert)

            # Execute custom handlers
            await self._execute_custom_handlers(alert)

    async def _send_email_alert(self, alert):
        """Send email alert."""
        if not self.email_config:
            return

        try:
            msg = MIMEText(f"Performance Alert: {alert['message']}")
            msg['Subject'] = f"CRITICAL: {alert['metric']} Alert"
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']

            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _execute_custom_handlers(self, alert):
        """Execute custom alert handlers."""
        # Example: Reduce concurrency on critical alerts
        if alert['level'] == 'CRITICAL' and alert['metric'] == 'memory_usage_mb':
            logger.info("Executing memory pressure handler")
            # Add custom handling logic here
```

## Visualization and Reporting

### Metrics Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd

class MetricsVisualizer:
    def __init__(self, metrics_data):
        self.metrics_data = metrics_data

    def plot_memory_usage(self):
        """Plot memory usage over time."""
        df = pd.DataFrame(self.metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['memory_usage_mb'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_execution_time(self):
        """Plot execution time over time."""
        df = pd.DataFrame(self.metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['execution_time_ms'])
        plt.title('Execution Time Over Time')
        plt.xlabel('Time')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_error_rate(self):
        """Plot error rate over time."""
        df = pd.DataFrame(self.metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['error_rate'])
        plt.title('Error Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self):
        """Generate a summary report."""
        df = pd.DataFrame(self.metrics_data)

        summary = {
            'memory_usage': {
                'mean': df['memory_usage_mb'].mean(),
                'max': df['memory_usage_mb'].max(),
                'min': df['memory_usage_mb'].min(),
            },
            'execution_time': {
                'mean': df['execution_time_ms'].mean(),
                'max': df['execution_time_ms'].max(),
                'min': df['execution_time_ms'].min(),
            },
            'error_rate': {
                'mean': df['error_rate'].mean(),
                'max': df['error_rate'].max(),
                'min': df['error_rate'].min(),
            },
            'cache_hit_rate': {
                'mean': df['cache_hit_rate'].mean(),
                'max': df['cache_hit_rate'].max(),
                'min': df['cache_hit_rate'].min(),
            }
        }

        return summary
```

### Performance Report Generator

```python
class PerformanceReportGenerator:
    def __init__(self, metrics_data, baseline_metrics=None):
        self.metrics_data = metrics_data
        self.baseline_metrics = baseline_metrics

    def generate_report(self):
        """Generate a comprehensive performance report."""
        df = pd.DataFrame(self.metrics_data)

        report = {
            'summary': self._generate_summary(df),
            'trends': self._analyze_trends(df),
            'recommendations': self._generate_recommendations(df),
            'alerts': self._check_alerts(df),
        }

        return report

    def _generate_summary(self, df):
        """Generate performance summary."""
        return {
            'total_executions': len(df),
            'avg_memory_usage': df['memory_usage_mb'].mean(),
            'avg_execution_time': df['execution_time_ms'].mean(),
            'avg_error_rate': df['error_rate'].mean(),
            'avg_cache_hit_rate': df['cache_hit_rate'].mean(),
        }

    def _analyze_trends(self, df):
        """Analyze performance trends."""
        # Calculate trends using linear regression
        x = np.arange(len(df))

        trends = {}
        for column in ['memory_usage_mb', 'execution_time_ms', 'error_rate']:
            y = df[column].values
            slope = np.polyfit(x, y, 1)[0]
            trends[column] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'

        return trends

    def _generate_recommendations(self, df):
        """Generate performance recommendations."""
        recommendations = []

        # Memory recommendations
        if df['memory_usage_mb'].mean() > 500:
            recommendations.append("Consider reducing object pool size to lower memory usage")

        # Execution time recommendations
        if df['execution_time_ms'].mean() > 1000:
            recommendations.append("Investigate slow execution - check for bottlenecks")

        # Error rate recommendations
        if df['error_rate'].mean() > 0.1:
            recommendations.append("High error rate detected - investigate error sources")

        # Cache recommendations
        if df['cache_hit_rate'].mean() < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache size")

        return recommendations

    def _check_alerts(self, df):
        """Check for performance alerts."""
        alerts = []

        # Check for anomalies
        for column in ['memory_usage_mb', 'execution_time_ms', 'error_rate']:
            mean = df[column].mean()
            std = df[column].std()

            # Check for values > 2 standard deviations from mean
            anomalies = df[df[column] > mean + 2 * std]
            if len(anomalies) > 0:
                alerts.append(f"Anomalies detected in {column}: {len(anomalies)} instances")

        return alerts
```

## Best Practices

### 1. Set Appropriate Thresholds
- Base thresholds on your specific workload
- Start conservative and adjust based on observations
- Consider both warning and critical thresholds

### 2. Monitor Continuously
- Set up continuous monitoring from the start
- Use multiple monitoring approaches (logs, metrics, alerts)
- Keep historical data for trend analysis

### 3. Respond to Alerts Quickly
- Have clear escalation procedures
- Document common issues and solutions
- Test alert mechanisms regularly

### 4. Use Multiple Metrics
- Don't rely on single metrics
- Correlate different performance indicators
- Consider system-wide impact

### 5. Document Performance Baselines
- Establish performance baselines
- Track performance changes over time
- Document optimization impacts

### 6. Regular Performance Reviews
- Schedule regular performance reviews
- Analyze trends and patterns
- Update monitoring strategies

## Conclusion

Effective performance monitoring for ExecutorCore optimizations requires a comprehensive approach that includes real-time monitoring, trend analysis, alerting, and visualization. By following the practices outlined in this guide, you can maintain optimal performance and quickly identify and resolve issues.

Remember that monitoring is not just about collecting metrics, but about using that data to make informed decisions about optimization strategies and system health.
