# ExecutorCore Performance Monitoring and Troubleshooting Guide

## Overview

This guide provides comprehensive instructions for monitoring ExecutorCore performance, setting up alerting systems, and troubleshooting performance issues. It includes practical tools, scripts, and methodologies developed during the optimization initiative.

## Performance Monitoring Setup

### 1. Built-in Performance Monitoring

#### Enable Performance Monitoring
```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

config = OptimizationConfig(
    enable_performance_monitoring=True,
    performance_monitoring_interval=300,  # 5 minutes
    performance_alert_threshold=0.2,      # 20% degradation alert
    performance_history_size=1000,        # Keep 1000 measurements
    performance_metrics_export=True,      # Export metrics for external systems
)

executor = ExecutorCore(optimization_config=config)
```

#### Access Performance Metrics
```python
# Get current performance metrics
metrics = executor.get_performance_metrics()
print(f"Current execution time: {metrics['execution_time_ms']:.2f}ms")
print(f"Memory usage: {metrics['memory_usage_mb']:.2f}MB")
print(f"Throughput: {metrics['throughput_ops_per_sec']:.1f} ops/sec")

# Get performance history
history = executor.get_performance_history()
for timestamp, metric_data in history:
    print(f"{timestamp}: {metric_data}")
```

### 2. Custom Performance Validation

#### Using the Performance Validation Script
```bash
# Run comprehensive performance validation
python scripts/performance_validation.py

# Run with custom configuration
python scripts/performance_validation.py --config custom_config.py

# Run specific test suites
python scripts/performance_validation.py --tests execution_performance,memory_efficiency

# Export results to file
python scripts/performance_validation.py --output performance_results.json
```

#### Performance Validation in Code
```python
from scripts.performance_validation import (
    run_performance_validation,
    compare_performance_results,
    generate_performance_report
)

# Run validation
results = run_performance_validation(
    use_optimized=True,
    config=your_config,
    iterations=100
)

# Compare with baseline
baseline_results = load_baseline_results()
comparison = compare_performance_results(baseline_results, results)

# Generate report
report = generate_performance_report(comparison)
print(report)
```

### 3. Telemetry Integration

#### Configure Telemetry Collection
```python
config = OptimizationConfig(
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=0.1,      # 10% sampling
    telemetry_batch_size=100,         # Batch size for efficiency
    telemetry_export_interval=60,     # Export every minute
    telemetry_retention_days=30       # Keep 30 days of data
)
```

#### Custom Telemetry Handlers
```python
from flujo.application.core.runtime.optimized_telemetry import TelemetryHandler

class CustomTelemetryHandler(TelemetryHandler):
    def handle_metric(self, metric_name, value, timestamp, tags=None):
        # Send to your monitoring system (e.g., Prometheus, DataDog, etc.)
        self.send_to_monitoring_system(metric_name, value, timestamp, tags)

    def handle_batch(self, metrics_batch):
        # Handle batch of metrics for efficiency
        self.send_batch_to_monitoring_system(metrics_batch)

# Register custom handler
executor.register_telemetry_handler(CustomTelemetryHandler())
```

## Key Performance Metrics

### 1. Execution Metrics
```python
# Core execution performance
execution_metrics = {
    'mean_execution_time_ms': 0.060,      # Average execution time
    'p95_execution_time_ms': 0.105,       # 95th percentile
    'p99_execution_time_ms': 0.328,       # 99th percentile
    'execution_throughput_ops_sec': 16659, # Operations per second
    'execution_success_rate': 1.0         # Success rate (0.0-1.0)
}
```

### 2. Memory Metrics
```python
# Memory usage and efficiency
memory_metrics = {
    'current_memory_usage_mb': 0.14,      # Current memory usage
    'peak_memory_usage_mb': 0.75,         # Peak memory usage
    'memory_growth_rate_mb_sec': 0.001,   # Memory growth rate
    'gc_frequency_per_min': 2.5,          # Garbage collection frequency
    'memory_efficiency_score': 0.85       # Efficiency score (0.0-1.0)
}
```

### 3. Concurrency Metrics
```python
# Concurrent execution performance
concurrency_metrics = {
    'concurrent_execution_time_ms': 0.755, # Concurrent operation time
    'active_threads': 12,                   # Currently active threads
    'thread_pool_utilization': 0.6,        # Thread pool utilization
    'lock_contention_rate': 0.02,          # Lock contention rate
    'deadlock_count': 0                     # Deadlock occurrences
}
```

### 4. Cache Metrics
```python
# Cache performance and efficiency
cache_metrics = {
    'cache_hit_rate': 0.85,               # Cache hit rate (0.0-1.0)
    'cache_miss_rate': 0.15,              # Cache miss rate
    'cache_size_mb': 2.5,                 # Current cache size
    'cache_eviction_rate': 0.1,           # Eviction rate
    'cache_lookup_time_ms': 0.116         # Average lookup time
}
```

### 5. Context Metrics
```python
# Context handling performance
context_metrics = {
    'context_creation_time_ms': 0.052,    # Context creation time
    'context_copy_time_ms': 0.025,        # Context copying time
    'context_size_bytes': 1024,           # Average context size
    'context_cache_hit_rate': 0.7,        # Context cache hit rate
    'context_serialization_time_ms': 0.1  # Serialization time
}
```

## Alerting and Thresholds

### 1. Performance Degradation Alerts

#### Critical Thresholds
```python
CRITICAL_THRESHOLDS = {
    'execution_time_regression': 0.5,     # 50% slower than baseline
    'memory_usage_increase': 2.0,         # 2x memory usage increase
    'throughput_decrease': 0.3,           # 30% throughput decrease
    'error_rate_increase': 0.05,          # 5% error rate
    'cache_hit_rate_decrease': 0.2        # 20% cache hit rate decrease
}
```

#### Warning Thresholds
```python
WARNING_THRESHOLDS = {
    'execution_time_regression': 0.2,     # 20% slower than baseline
    'memory_usage_increase': 1.5,         # 1.5x memory usage increase
    'throughput_decrease': 0.15,          # 15% throughput decrease
    'error_rate_increase': 0.02,          # 2% error rate
    'cache_hit_rate_decrease': 0.1        # 10% cache hit rate decrease
}
```

#### Alert Implementation
```python
class PerformanceAlerter:
    def __init__(self, critical_thresholds, warning_thresholds):
        self.critical_thresholds = critical_thresholds
        self.warning_thresholds = warning_thresholds
        self.baseline_metrics = self.load_baseline_metrics()

    def check_metrics(self, current_metrics):
        alerts = []

        for metric_name, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric_name)
            if baseline_value is None:
                continue

            regression = (current_value - baseline_value) / baseline_value

            if regression > self.critical_thresholds.get(metric_name, float('inf')):
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': metric_name,
                    'current': current_value,
                    'baseline': baseline_value,
                    'regression': regression
                })
            elif regression > self.warning_thresholds.get(metric_name, float('inf')):
                alerts.append({
                    'level': 'WARNING',
                    'metric': metric_name,
                    'current': current_value,
                    'baseline': baseline_value,
                    'regression': regression
                })

        return alerts
```

### 2. System Resource Monitoring

#### CPU Monitoring
```python
import psutil

def monitor_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

    return {
        'cpu_total_percent': cpu_percent,
        'cpu_per_core': cpu_per_core,
        'cpu_count': psutil.cpu_count(),
        'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }
```

#### Memory Monitoring
```python
def monitor_memory_usage():
    memory = psutil.virtual_memory()

    return {
        'memory_total_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent,
        'swap_total_gb': psutil.swap_memory().total / (1024**3),
        'swap_used_gb': psutil.swap_memory().used / (1024**3)
    }
```

## Troubleshooting Guide

### 1. Performance Regression Troubleshooting

#### Step 1: Identify the Regression
```python
# Run performance comparison
current_results = run_performance_validation(use_optimized=True)
baseline_results = load_baseline_results()

# Identify regressed metrics
regressions = identify_regressions(baseline_results, current_results)
for metric, regression_pct in regressions.items():
    print(f"{metric}: {regression_pct:.1f}% regression")
```

#### Step 2: Isolate the Cause
```python
# Test with optimizations disabled
no_opt_results = run_performance_validation(use_optimized=False)

# Test individual optimizations
optimization_impact = {}
for opt_name in ['object_pool', 'context_optimization', 'memory_optimization']:
    config = create_single_optimization_config(opt_name)
    results = run_performance_validation(use_optimized=True, config=config)
    optimization_impact[opt_name] = calculate_impact(baseline_results, results)

# Identify problematic optimization
problematic_opts = [opt for opt, impact in optimization_impact.items() if impact < -0.1]
print(f"Problematic optimizations: {problematic_opts}")
```

#### Step 3: Parameter Tuning
```python
# Tune parameters for problematic optimization
if 'object_pool' in problematic_opts:
    # Test different pool sizes
    pool_sizes = [50, 100, 250, 500, 1000, 2000]
    best_size = tune_object_pool_size(pool_sizes)
    print(f"Optimal pool size: {best_size}")

if 'context_optimization' in problematic_opts:
    # Test different cache sizes
    cache_sizes = [256, 512, 1024, 2048, 4096]
    best_cache_size = tune_context_cache_size(cache_sizes)
    print(f"Optimal cache size: {best_cache_size}")
```

### 2. Memory Issues Troubleshooting

#### Memory Leak Detection
```python
import tracemalloc

def detect_memory_leaks():
    tracemalloc.start()

    # Run your workload
    for i in range(1000):
        run_single_operation()

        if i % 100 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")

    # Get top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("Top 10 memory consumers:")
    for stat in top_stats[:10]:
        print(stat)
```

#### Memory Usage Analysis
```python
def analyze_memory_usage():
    import gc
    import sys

    # Force garbage collection
    gc.collect()

    # Get object counts by type
    object_counts = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

    # Sort by count
    sorted_counts = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)

    print("Top object types by count:")
    for obj_type, count in sorted_counts[:20]:
        print(f"{obj_type}: {count}")

    # Get total memory usage
    total_size = sys.getsizeof(gc.get_objects())
    print(f"Total tracked object size: {total_size / 1024 / 1024:.1f}MB")
```

### 3. Concurrency Issues Troubleshooting

#### Deadlock Detection
```python
import threading
import time

class DeadlockDetector:
    def __init__(self):
        self.lock_owners = {}
        self.lock_waiters = {}
        self.detection_interval = 5.0

    def register_lock_acquisition(self, lock_id, thread_id):
        self.lock_owners[lock_id] = thread_id

    def register_lock_wait(self, lock_id, thread_id):
        if lock_id not in self.lock_waiters:
            self.lock_waiters[lock_id] = []
        self.lock_waiters[lock_id].append(thread_id)

    def detect_deadlocks(self):
        # Simple cycle detection in lock dependency graph
        for lock_id, waiters in self.lock_waiters.items():
            owner = self.lock_owners.get(lock_id)
            if owner and owner in waiters:
                print(f"Potential deadlock detected: Lock {lock_id}")
                return True
        return False
```

#### Thread Pool Analysis
```python
def analyze_thread_pool():
    import concurrent.futures

    # Monitor thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks and monitor
        futures = []
        for i in range(100):
            future = executor.submit(sample_task, i)
            futures.append(future)

        # Monitor completion
        completed = 0
        start_time = time.time()

        for future in concurrent.futures.as_completed(futures):
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed
            print(f"Completed: {completed}/100, Rate: {rate:.1f} tasks/sec")
```

### 4. Cache Performance Troubleshooting

#### Cache Hit Rate Analysis
```python
def analyze_cache_performance():
    cache_stats = executor.get_cache_statistics()

    print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"Cache miss rate: {cache_stats['miss_rate']:.2%}")
    print(f"Cache size: {cache_stats['size']} items")
    print(f"Cache memory usage: {cache_stats['memory_usage_mb']:.1f}MB")

    # Analyze cache key patterns
    key_patterns = analyze_cache_keys(cache_stats['keys'])
    print("Most common key patterns:")
    for pattern, count in key_patterns.items():
        print(f"  {pattern}: {count} keys")
```

#### Cache Eviction Analysis
```python
def analyze_cache_evictions():
    eviction_stats = executor.get_cache_eviction_statistics()

    print(f"Total evictions: {eviction_stats['total_evictions']}")
    print(f"Eviction rate: {eviction_stats['eviction_rate']:.2f}/sec")

    # Analyze eviction reasons
    for reason, count in eviction_stats['eviction_reasons'].items():
        print(f"  {reason}: {count} evictions")

    # Analyze evicted key patterns
    evicted_patterns = analyze_evicted_keys(eviction_stats['evicted_keys'])
    print("Most evicted key patterns:")
    for pattern, count in evicted_patterns.items():
        print(f"  {pattern}: {count} evictions")
```

## Diagnostic Tools

### 1. Performance Profiler
```python
import cProfile
import pstats
import io

def profile_executor_performance():
    profiler = cProfile.Profile()

    # Profile the execution
    profiler.enable()

    # Run your workload
    for i in range(100):
        executor.execute(sample_workflow)

    profiler.disable()

    # Analyze results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)

    profile_output = s.getvalue()
    print(profile_output)

    return profile_output
```

### 2. Memory Profiler
```python
from memory_profiler import profile

@profile
def memory_profile_executor():
    # Create executor
    executor = UltraExecutor(config)

    # Run workload
    for i in range(100):
        result = executor.execute(sample_workflow)

    # Clean up
    del executor
```

### 3. Performance Regression Detector
```python
class PerformanceRegressionDetector:
    def __init__(self, baseline_file='baseline_performance.json'):
        self.baseline_file = baseline_file
        self.baseline_metrics = self.load_baseline()

    def load_baseline(self):
        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_baseline(self, metrics):
        with open(self.baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def detect_regressions(self, current_metrics, threshold=0.1):
        regressions = []

        for metric_name, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric_name)
            if baseline_value is None:
                continue

            if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                change = (current_value - baseline_value) / baseline_value
                if abs(change) > threshold:
                    regressions.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_percent': change * 100,
                        'regression': change > 0  # Assuming higher is worse
                    })

        return regressions
```

## Monitoring Dashboard Setup

### 1. Metrics Collection Script
```python
#!/usr/bin/env python3
"""
Performance metrics collection script for ExecutorCore monitoring.
Run this script periodically to collect and export performance metrics.
"""

import json
import time
import argparse
from datetime import datetime, timezone
from scripts.performance_validation import run_performance_validation

def collect_metrics(config_file=None):
    """Collect current performance metrics."""

    # Load configuration
    config = load_config(config_file) if config_file else None

    # Run performance validation
    results = run_performance_validation(
        use_optimized=bool(config),
        config=config,
        iterations=10  # Fewer iterations for monitoring
    )

    # Add timestamp and system info
    results['timestamp'] = datetime.now(timezone.utc).isoformat()
    results['system_info'] = get_system_info()

    return results

def export_metrics(metrics, output_format='json'):
    """Export metrics in specified format."""

    if output_format == 'json':
        print(json.dumps(metrics, indent=2))
    elif output_format == 'prometheus':
        export_prometheus_metrics(metrics)
    elif output_format == 'csv':
        export_csv_metrics(metrics)

def main():
    parser = argparse.ArgumentParser(description='Collect ExecutorCore performance metrics')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--format', choices=['json', 'prometheus', 'csv'],
                       default='json', help='Output format')
    parser.add_argument('--interval', type=int, help='Collection interval in seconds')

    args = parser.parse_args()

    if args.interval:
        # Continuous collection
        while True:
            metrics = collect_metrics(args.config)
            export_metrics(metrics, args.format)
            time.sleep(args.interval)
    else:
        # Single collection
        metrics = collect_metrics(args.config)
        export_metrics(metrics, args.format)

if __name__ == '__main__':
    main()
```

### 2. Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "ExecutorCore Performance",
    "panels": [
      {
        "title": "Execution Time",
        "type": "graph",
        "targets": [
          {
            "expr": "executor_execution_time_ms",
            "legendFormat": "Execution Time (ms)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "executor_memory_usage_mb",
            "legendFormat": "Memory Usage (MB)"
          }
        ]
      },
      {
        "title": "Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "executor_throughput_ops_per_sec",
            "legendFormat": "Operations/sec"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "executor_cache_hit_rate",
            "legendFormat": "Hit Rate"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Monitoring Strategy
- **Baseline Establishment**: Always establish baseline metrics before optimization
- **Continuous Monitoring**: Monitor key metrics continuously in production
- **Alert Tuning**: Tune alert thresholds based on actual performance patterns
- **Historical Analysis**: Keep historical data for trend analysis

### 2. Troubleshooting Approach
- **Systematic Investigation**: Follow a systematic approach to isolate issues
- **Data-Driven Decisions**: Base decisions on actual performance data
- **Incremental Changes**: Make incremental changes and validate each step
- **Documentation**: Document findings and solutions for future reference

### 3. Performance Testing
- **Realistic Workloads**: Use realistic workloads for performance testing
- **Multiple Scenarios**: Test various scenarios and edge cases
- **Statistical Significance**: Ensure statistical significance in measurements
- **Environment Consistency**: Maintain consistent test environments

## Conclusion

Effective monitoring and troubleshooting of ExecutorCore performance requires:

1. **Comprehensive Metrics Collection**: Monitor all key performance indicators
2. **Proactive Alerting**: Set up alerts for performance degradation
3. **Systematic Troubleshooting**: Follow structured approaches to problem resolution
4. **Continuous Improvement**: Use monitoring data to drive optimization decisions

The tools and methodologies provided in this guide enable effective performance management and rapid issue resolution for ExecutorCore deployments.

---

*Guide Version: 1.0*
*Last Updated: July 31, 2025*
*Based on: ExecutorCore Optimization Performance Analysis*
