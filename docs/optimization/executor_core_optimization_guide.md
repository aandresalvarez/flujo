# ExecutorCore Optimization Guide

## Overview

The ExecutorCore optimization system provides comprehensive performance improvements for Flujo pipelines through memory optimization, execution optimization, telemetry optimization, and error handling optimization. This guide covers how to configure and use these optimizations effectively.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Optimization Components](#optimization-components)
3. [Configuration Guide](#configuration-guide)
4. [Performance Tuning](#performance-tuning)
5. [Use Case Examples](#use-case-examples)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring and Metrics](#monitoring-and-metrics)

## Quick Start

### Basic Usage

> Recommended entrypoint: use `ExecutorCore` with an `OptimizationConfig`. The legacy `OptimizedExecutorCore` shim has been removed; update imports accordingly.

```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

# Create optimized executor with default settings
executor = ExecutorCore(optimization_config=OptimizationConfig())

# Execute a step with optimizations
result = await executor.execute(step, data, context=context)
```

### Custom Configuration

```python
# Create custom optimization configuration
config = OptimizationConfig(
    enable_object_pool=True,
    enable_step_optimization=True,
    enable_algorithm_optimization=True,
    max_concurrent_executions=10
)

# Create executor with custom configuration
executor = ExecutorCore(optimization_config=config)
```

## Optimization Components

### 1. Memory Optimizations

#### Object Pooling
- **Purpose**: Reduces memory allocation overhead by reusing objects
- **Components**: `OptimizedObjectPool`
- **Benefits**: 20-40% reduction in memory allocation for frequently used objects

#### Context Management
- **Purpose**: Optimizes context copying and merging operations
- **Components**: `OptimizedContextManager`
- **Benefits**: Copy-on-write optimization and intelligent caching

#### Memory Pressure Detection
- **Purpose**: Automatically adjusts memory usage based on system pressure
- **Components**: `AdaptiveResourceManager`
- **Benefits**: Prevents out-of-memory errors and maintains performance

### 2. Execution Optimizations

#### Step Execution Optimization
- **Purpose**: Pre-analyzes steps and caches analysis results
- **Components**: `OptimizedStepExecutor`
- **Benefits**: Faster step execution through cached analysis

#### Algorithm Optimizations
- **Purpose**: Optimizes hashing, serialization, and cache key generation
- **Components**: `AlgorithmOptimizations`
- **Benefits**: Faster cache operations and reduced CPU usage

#### Concurrency Optimization
- **Purpose**: Adaptive concurrency limits and work-stealing queues
- **Components**: `ConcurrencyOptimizations`
- **Benefits**: Better resource utilization and reduced contention

### 3. Telemetry Optimizations

#### Low-Overhead Tracing
- **Purpose**: Minimal performance impact telemetry collection
- **Components**: `OptimizedTelemetry`
- **Benefits**: Real-time monitoring without performance degradation

#### Performance Monitoring
- **Purpose**: Real-time performance metrics and threshold detection
- **Components**: `PerformanceMonitor`
- **Benefits**: Automatic performance regression detection

### 4. Error Handling Optimizations

#### Error Caching
- **Purpose**: Caches error patterns for faster recovery
- **Components**: `OptimizedErrorHandler`
- **Benefits**: Faster error recovery and reduced repeated failures

#### Circuit Breaker
- **Purpose**: Prevents cascade failures through automatic failure detection
- **Components**: `CircuitBreaker`
- **Benefits**: Improved system resilience and automatic recovery

## Configuration Guide

### OptimizationConfig Parameters

#### Memory Optimization Settings

```python
config = OptimizationConfig(
    # Object pooling settings
    enable_object_pool=True,
    object_pool_max_size=50,
    object_pool_cleanup_threshold=0.9,

    # Context optimization
    enable_context_optimization=True,
    enable_memory_optimization=True,
)
```

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `enable_object_pool` | `True` | Enable object pooling for memory reuse | 20-40% memory reduction |
| `object_pool_max_size` | `50` | Maximum objects per type in pool | Higher = more memory, faster allocation |
| `object_pool_cleanup_threshold` | `0.9` | When to clean up pool (0.0-1.0) | Lower = more aggressive cleanup |
| `enable_context_optimization` | `True` | Enable optimized context handling | 15-25% context operation speedup |
| `enable_memory_optimization` | `True` | Enable memory pressure detection | Automatic memory management |

#### Execution Optimization Settings

```python
config = OptimizationConfig(
    # Step execution optimization
    enable_step_optimization=True,
    enable_algorithm_optimization=True,
    enable_concurrency_optimization=True,
    max_concurrent_executions=None,  # Auto-detect
)
```

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `enable_step_optimization` | `True` | Enable step analysis caching | 10-30% step execution speedup |
| `enable_algorithm_optimization` | `True` | Enable optimized hashing/serialization | 20-50% cache operation speedup |
| `enable_concurrency_optimization` | `True` | Enable adaptive concurrency | Better resource utilization |
| `max_concurrent_executions` | `None` | Maximum concurrent executions | Auto-detect if None |

#### Telemetry Optimization Settings

```python
config = OptimizationConfig(
    # Telemetry settings
    enable_optimized_telemetry=True,
    enable_performance_monitoring=True,
    telemetry_batch_size=10,
    telemetry_flush_interval_seconds=30.0,
)
```

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `enable_optimized_telemetry` | `True` | Enable low-overhead telemetry | <1% performance impact |
| `enable_performance_monitoring` | `True` | Enable performance monitoring | Real-time metrics |
| `telemetry_batch_size` | `10` | Number of metrics per batch | Higher = less overhead |
| `telemetry_flush_interval_seconds` | `30.0` | How often to flush metrics | Higher = less frequent I/O |

#### Error Handling Optimization Settings

```python
config = OptimizationConfig(
    # Error handling settings
    enable_optimized_error_handling=True,
    enable_circuit_breaker=True,
    error_cache_size=50,
    circuit_breaker_failure_threshold=10,
    circuit_breaker_recovery_timeout_seconds=60,
)
```

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `enable_optimized_error_handling` | `True` | Enable error pattern caching | Faster error recovery |
| `enable_circuit_breaker` | `True` | Enable circuit breaker pattern | Prevents cascade failures |
| `error_cache_size` | `50` | Number of cached error patterns | Higher = more memory, faster recovery |
| `circuit_breaker_failure_threshold` | `10` | Failures before opening circuit | Lower = more sensitive |
| `circuit_breaker_recovery_timeout_seconds` | `60` | Time before attempting recovery | Higher = longer recovery time |

#### Cache Optimization Settings

```python
config = OptimizationConfig(
    # Cache settings (disabled by default)
    enable_cache_optimization=False,
    cache_compression=False,
    cache_ttl_seconds=7200,
    cache_max_size=500,
)
```

| Parameter | Default | Description | Performance Impact |
|-----------|---------|-------------|-------------------|
| `enable_cache_optimization` | `False` | Enable advanced cache optimizations | May add overhead for simple operations |
| `cache_compression` | `False` | Enable cache compression | Higher CPU, lower memory |
| `cache_ttl_seconds` | `7200` | Cache entry time-to-live | Higher = longer cache retention |
| `cache_max_size` | `500` | Maximum cache entries | Higher = more memory, better hit rate |

#### Performance Thresholds

```python
config = OptimizationConfig(
    # Performance thresholds
    slow_execution_threshold_ms=1000.0,
    memory_pressure_threshold_mb=500.0,
    cpu_usage_threshold_percent=80.0,
)
```

| Parameter | Default | Description | Usage |
|-----------|---------|-------------|-------|
| `slow_execution_threshold_ms` | `1000.0` | Threshold for slow execution detection | Performance monitoring |
| `memory_pressure_threshold_mb` | `500.0` | Memory pressure threshold | Automatic memory management |
| `cpu_usage_threshold_percent` | `80.0` | CPU usage threshold | Resource management |

### Configuration Validation

```python
# Validate configuration
config = OptimizationConfig()
issues = config.validate()

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Configuration is valid")
```

## Performance Tuning

### Workload-Specific Configurations

#### High-Throughput Workloads

```python
# Optimized for high throughput
config = OptimizationConfig(
    # Memory optimizations
    object_pool_max_size=100,
    object_pool_cleanup_threshold=0.8,

    # Execution optimizations
    max_concurrent_executions=20,

    # Telemetry optimizations
    telemetry_batch_size=50,
    telemetry_flush_interval_seconds=60.0,

    # Cache optimizations
    enable_cache_optimization=True,
    cache_max_size=1000,
    cache_ttl_seconds=3600,
)
```

#### Memory-Constrained Environments

```python
# Optimized for memory-constrained environments
config = OptimizationConfig(
    # Memory optimizations
    object_pool_max_size=20,
    object_pool_cleanup_threshold=0.7,

    # Cache optimizations
    cache_max_size=200,
    cache_compression=True,

    # Performance thresholds
    memory_pressure_threshold_mb=200.0,
)
```

#### Latency-Sensitive Applications

```python
# Optimized for low latency
config = OptimizationConfig(
    # Execution optimizations
    enable_step_optimization=True,
    enable_algorithm_optimization=True,

    # Telemetry optimizations
    telemetry_batch_size=5,
    telemetry_flush_interval_seconds=10.0,

    # Error handling
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout_seconds=30,
)
```

### Automatic Optimization

```python
# Enable automatic optimization
config = OptimizationConfig(
    enable_automatic_optimization=True,
    optimization_analysis_interval_seconds=60.0,
    performance_degradation_threshold=0.2,
)

executor = ExecutorCore(optimization_config=config)

# The executor will automatically adjust settings based on performance
```

## Use Case Examples

### Example 1: Data Processing Pipeline

```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

# Configuration for data processing
config = OptimizationConfig(
    # Memory optimizations for large datasets
    object_pool_max_size=100,
    enable_memory_optimization=True,

    # Execution optimizations for repeated operations
    enable_step_optimization=True,
    enable_algorithm_optimization=True,

    # Concurrency for parallel processing
    max_concurrent_executions=10,

    # Cache for repeated operations
    enable_cache_optimization=True,
    cache_max_size=1000,
)

executor = ExecutorCore(optimization_config=config)

# Process large dataset
for batch in data_batches:
    result = await executor.execute(process_batch_step, batch)
```

### Example 2: Real-Time API Service

```python
# Configuration for real-time API
config = OptimizationConfig(
    # Low latency optimizations
    enable_step_optimization=True,
    enable_algorithm_optimization=True,

    # Minimal telemetry overhead
    telemetry_batch_size=5,
    telemetry_flush_interval_seconds=10.0,

    # Fast error recovery
    circuit_breaker_failure_threshold=3,
    circuit_breaker_recovery_timeout_seconds=30,

    # Minimal memory usage
    object_pool_max_size=20,
    cache_max_size=100,
)

executor = ExecutorCore(optimization_config=config)

# Handle API requests
async def handle_request(request_data):
    return await executor.execute(api_step, request_data)
```

### Example 3: Batch Processing with Monitoring

```python
# Configuration with monitoring
config = OptimizationConfig(
    enable_performance_monitoring=True,
    enable_optimized_telemetry=True,
    enable_automatic_optimization=True,
)

executor = ExecutorCore(optimization_config=config)

# Execute with monitoring
result, metrics = await executor.execute_with_monitoring(
    batch_process_step,
    batch_data
)

print(f"Execution time: {metrics['execution_time_ms']}ms")
print(f"Memory usage: {metrics['memory_usage_mb']}MB")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: High Memory Usage
**Symptoms**: Memory usage grows over time, potential out-of-memory errors

**Solutions**:
```python
# Reduce object pool size
config = OptimizationConfig(
    object_pool_max_size=20,
    object_pool_cleanup_threshold=0.6,
)

# Enable memory pressure detection
config.enable_memory_optimization = True
```

#### Issue: Slow Execution
**Symptoms**: Step execution takes longer than expected

**Solutions**:
```python
# Enable step optimization
config = OptimizationConfig(
    enable_step_optimization=True,
    enable_algorithm_optimization=True,
)

# Check performance recommendations
recommendations = executor.get_performance_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec['description']}")
```

#### Issue: High CPU Usage
**Symptoms**: CPU usage spikes during execution

**Solutions**:
```python
# Reduce telemetry overhead
config = OptimizationConfig(
    telemetry_batch_size=50,
    telemetry_flush_interval_seconds=60.0,
)

# Limit concurrency
config.max_concurrent_executions = 5
```

#### Issue: Frequent Errors
**Symptoms**: High error rate, cascade failures

**Solutions**:
```python
# Enable circuit breaker
config = OptimizationConfig(
    enable_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout_seconds=30,
)

# Enable error caching
config.enable_optimized_error_handling = True
```

### Debugging Configuration

```python
# Get current configuration
current_config = executor.optimization_config
print(f"Current config: {current_config.to_dict()}")

# Get optimization statistics
stats = executor.get_optimization_stats()
print(f"Optimization stats: {stats}")

# Get performance recommendations
recommendations = executor.get_performance_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

### Performance Monitoring

```python
# Monitor performance in real-time
async def monitor_performance():
    while True:
        stats = executor.get_optimization_stats()

        print(f"Memory usage: {stats['memory']['usage_mb']}MB")
        print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
        print(f"Average execution time: {stats['execution']['avg_time_ms']}ms")

        await asyncio.sleep(10)

# Start monitoring
asyncio.create_task(monitor_performance())
```

## Monitoring and Metrics

### Available Metrics

#### Memory Metrics
- `object_pool_utilization`: Object pool usage percentage
- `memory_usage_mb`: Current memory usage
- `memory_pressure_level`: Memory pressure indicator

#### Performance Metrics
- `execution_time_ms`: Step execution time
- `cache_hit_rate`: Cache hit rate percentage
- `concurrency_level`: Current concurrency level

#### Error Metrics
- `error_rate`: Error rate percentage
- `circuit_breaker_state`: Circuit breaker state
- `recovery_success_rate`: Error recovery success rate

### Exporting Configuration

```python
# Export configuration to JSON
config_json = executor.export_config(format="json")
with open("optimization_config.json", "w") as f:
    f.write(config_json)

# Import configuration
with open("optimization_config.json", "r") as f:
    config_data = f.read()
await executor.import_config(config_data, format="json")
```

### Performance Recommendations

```python
# Get automatic performance recommendations
recommendations = executor.get_performance_recommendations()

for rec in recommendations:
    print(f"Recommendation: {rec['description']}")
    print(f"Impact: {rec['impact']}")
    print(f"Priority: {rec['priority']}")

    if rec['automatic']:
        print("This can be applied automatically")
    else:
        print("Manual configuration required")
```

## Best Practices

### 1. Start with Default Configuration
Begin with the default configuration and adjust based on your specific needs:

```python
executor = ExecutorCore(optimization_config=OptimizationConfig())  # Use defaults first
```

### 2. Monitor Performance
Regularly monitor performance metrics to identify optimization opportunities:

```python
stats = executor.get_optimization_stats()
recommendations = executor.get_performance_recommendations()
```

### 3. Use Automatic Optimization
Enable automatic optimization for dynamic environments:

```python
config = OptimizationConfig(enable_automatic_optimization=True)
```

### 4. Validate Configuration
Always validate your configuration before deployment:

```python
issues = config.validate()
if issues:
    # Fix configuration issues
    pass
```

### 5. Test with Real Workloads
Test optimization settings with your actual workload patterns:

```python
# Test with representative data
test_results = await executor.execute_with_monitoring(test_step, test_data)
```

## Migration Guide

### Legacy note: OptimizedExecutorCore (removed)

```python
# Old code (removed)
# from flujo.application.core.executor_core import OptimizedExecutorCore
# executor = OptimizedExecutorCore()

# Supported code
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
executor = ExecutorCore(optimization_config=OptimizationConfig())
```

## Conclusion

The ExecutorCore optimization system provides significant performance improvements while maintaining ease of use and backward compatibility. By following this guide and monitoring your application's performance, you can achieve optimal performance for your specific use case.

For additional support or questions, refer to the test suite examples and benchmark results for more detailed performance characteristics.
