# Optimization Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered when using ExecutorCore optimizations. Each section includes symptoms, root causes, and step-by-step solutions.

## Table of Contents

1. [Performance Issues](#performance-issues)
2. [Memory Issues](#memory-issues)
3. [Configuration Issues](#configuration-issues)
4. [Error Handling Issues](#error-handling-issues)
5. [Monitoring Issues](#monitoring-issues)
6. [Debugging Tools](#debugging-tools)

## Performance Issues

### Issue: Performance Regression After Enabling Optimizations

**Symptoms:**
- Slower execution times after enabling optimizations
- Higher CPU usage than expected
- Increased latency in API responses

**Root Causes:**
- Overhead from optimization components exceeds benefits
- Inappropriate configuration for workload
- Resource contention from optimization features

**Solutions:**

#### Step 1: Identify the Problematic Component
```python
# Test each optimization component individually
from flujo.application.core.executor_core import OptimizationConfig

# Test baseline performance
baseline_executor = ExecutorCore()
baseline_time = await measure_execution_time(baseline_executor, test_step, test_data)

# Test with only memory optimizations
memory_config = OptimizationConfig(
    enable_object_pool=True,
    enable_context_optimization=True,
    enable_memory_optimization=True,
    # Disable all other optimizations
    enable_step_optimization=False,
    enable_algorithm_optimization=False,
    enable_concurrency_optimization=False,
    enable_optimized_telemetry=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
)
memory_executor = ExecutorCore(optimization_config=memory_config)
memory_time = await measure_execution_time(memory_executor, test_step, test_data)

print(f"Baseline: {baseline_time}ms")
print(f"Memory optimizations: {memory_time}ms")
print(f"Difference: {((memory_time - baseline_time) / baseline_time) * 100:.1f}%")
```

#### Step 2: Adjust Configuration Based on Workload
```python
# For CPU-intensive workloads, reduce telemetry overhead
cpu_optimized_config = OptimizationConfig(
    # Minimal telemetry
    telemetry_batch_size=100,
    telemetry_flush_interval_seconds=60.0,

    # Conservative concurrency
    max_concurrent_executions=4,

    # Disable heavy optimizations
    enable_cache_optimization=False,
    enable_automatic_optimization=False,
)
```

#### Step 3: Use Performance Monitoring
```python
# Monitor performance in real-time
async def monitor_performance():
    executor = ExecutorCore()

    while True:
        stats = executor.get_optimization_stats()

        if stats['execution']['avg_time_ms'] > 1000:  # 1 second threshold
            print("WARNING: Slow execution detected")
            recommendations = executor.get_performance_recommendations()
            for rec in recommendations:
                print(f"Recommendation: {rec['description']}")

        await asyncio.sleep(10)
```

### Issue: High CPU Usage

**Symptoms:**
- CPU usage spikes during execution
- System becomes unresponsive
- High CPU usage even during idle periods

**Root Causes:**
- Excessive telemetry collection
- Too many concurrent operations
- Inefficient algorithm optimizations

**Solutions:**

#### Reduce Telemetry Overhead
```python
# Minimize telemetry impact
config = OptimizationConfig(
    # Reduce telemetry frequency
    telemetry_batch_size=100,
    telemetry_flush_interval_seconds=60.0,

    # Disable performance monitoring if not needed
    enable_performance_monitoring=False,

    # Use minimal telemetry
    enable_optimized_telemetry=True,
)
```

#### Limit Concurrency
```python
# Reduce concurrent operations
config = OptimizationConfig(
    # Conservative concurrency
    max_concurrent_executions=2,  # Reduce from default

    # Disable concurrency optimization if causing issues
    enable_concurrency_optimization=False,
)
```

#### Disable Heavy Optimizations
```python
# Disable CPU-intensive optimizations
config = OptimizationConfig(
    # Disable algorithm optimizations if causing CPU spikes
    enable_algorithm_optimization=False,

    # Disable automatic optimization
    enable_automatic_optimization=False,

    # Disable cache optimization
    enable_cache_optimization=False,
)
```

### Issue: Inconsistent Performance

**Symptoms:**
- Variable execution times
- Performance varies between runs
- Unpredictable behavior

**Root Causes:**
- Memory pressure affecting performance
- Resource contention
- Automatic optimization conflicts

**Solutions:**

#### Stabilize Configuration
```python
# Use fixed configuration instead of automatic optimization
config = OptimizationConfig(
    # Disable automatic optimization
    enable_automatic_optimization=False,

    # Use fixed concurrency
    max_concurrent_executions=4,

    # Fixed cache settings
    cache_max_size=500,
    cache_ttl_seconds=3600,
)
```

#### Monitor Resource Usage
```python
# Monitor system resources
async def monitor_resources():
    import psutil

    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > 80 or memory_percent > 80:
            print(f"WARNING: High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")

            # Consider reducing optimizations
            if cpu_percent > 90:
                print("Consider disabling algorithm optimizations")
            if memory_percent > 90:
                print("Consider reducing object pool size")

        await asyncio.sleep(5)
```

## Memory Issues

### Issue: High Memory Usage

**Symptoms:**
- Memory usage grows over time
- Out-of-memory errors
- System becomes slow due to memory pressure

**Root Causes:**
- Object pools not being cleaned up
- Cache sizes too large
- Memory leaks in optimization components

**Solutions:**

#### Reduce Object Pool Size
```python
# Reduce object pool size
config = OptimizationConfig(
    # Smaller object pools
    object_pool_max_size=20,  # Reduce from default 50

    # More aggressive cleanup
    object_pool_cleanup_threshold=0.6,  # More aggressive than default 0.9
)
```

#### Reduce Cache Sizes
```python
# Reduce cache sizes
config = OptimizationConfig(
    # Smaller caches
    cache_max_size=100,  # Reduce from default 500
    error_cache_size=20,  # Reduce from default 50

    # Shorter TTLs
    cache_ttl_seconds=1800,  # Reduce from default 7200
)
```

#### Enable Memory Pressure Detection
```python
# Enable memory optimization
config = OptimizationConfig(
    # Enable memory pressure detection
    enable_memory_optimization=True,

    # Lower memory pressure threshold
    memory_pressure_threshold_mb=200.0,  # Reduce from default 500.0
)
```

### Issue: Memory Leaks

**Symptoms:**
- Memory usage never decreases
- Memory usage increases even during idle periods
- Out-of-memory errors after long running

**Root Causes:**
- Objects not being returned to pools
- Cache entries not expiring
- Circular references in optimization components

**Solutions:**

#### Force Cleanup
```python
# Force cleanup of optimization components
async def force_cleanup(executor):
    # Clear all caches
    await executor.clear_cache()

    # Clear object pools
    if hasattr(executor, '_object_pool'):
        executor._object_pool.clear()

    # Clear context caches
    if hasattr(executor, '_context_manager_opt'):
        executor._context_manager_opt.clear()
```

#### Monitor Memory Usage
```python
# Monitor memory usage and force cleanup when needed
async def memory_monitor(executor):
    import psutil

    while True:
        memory_percent = psutil.virtual_memory().percent

        if memory_percent > 85:
            print("WARNING: High memory usage, forcing cleanup")
            await force_cleanup(executor)

        await asyncio.sleep(30)
```

#### Use Memory-Optimized Configuration
```python
# Use memory-optimized configuration
config = OptimizationConfig(
    # Minimal object pools
    object_pool_max_size=10,
    object_pool_cleanup_threshold=0.5,

    # Small caches
    cache_max_size=50,
    error_cache_size=10,

    # Short TTLs
    cache_ttl_seconds=900,

    # Enable memory optimization
    enable_memory_optimization=True,
    memory_pressure_threshold_mb=150.0,
)
```

## Configuration Issues

### Issue: Invalid Configuration

**Symptoms:**
- Configuration validation errors
- Runtime errors during initialization
- Unexpected behavior

**Root Causes:**
- Invalid parameter values
- Conflicting settings
- Missing dependencies

**Solutions:**

#### Validate Configuration
```python
# Validate configuration before use
config = OptimizationConfig()

# Check for validation issues
issues = config.validate()
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")

    # Fix common issues
    if "object_pool_max_size must be positive" in issues:
        config.object_pool_max_size = 10

    if "telemetry_batch_size must be positive" in issues:
        config.telemetry_batch_size = 10

    # Re-validate
    issues = config.validate()
    if not issues:
        print("Configuration is now valid")
```

#### Use Safe Defaults
```python
# Use safe default configuration
safe_config = OptimizationConfig(
    # Conservative memory settings
    object_pool_max_size=20,
    object_pool_cleanup_threshold=0.7,

    # Conservative cache settings
    cache_max_size=100,
    error_cache_size=20,

    # Conservative concurrency
    max_concurrent_executions=4,

    # Minimal telemetry
    telemetry_batch_size=50,
    telemetry_flush_interval_seconds=60.0,

    # Conservative error handling
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout_seconds=30,
)
```

### Issue: Configuration Not Applied

**Symptoms:**
- Configuration changes not taking effect
- Old settings still being used
- Configuration appears to be ignored

**Root Causes:**
- Configuration not properly passed to executor
- Runtime configuration changes not supported
- Configuration validation failures

**Solutions:**

#### Verify Configuration Application
```python
# Verify configuration is applied
executor = ExecutorCore(optimization_config=config)

# Check current configuration
current_config = executor.optimization_config
print(f"Applied configuration: {current_config.to_dict()}")

# Verify specific settings
print(f"Object pool size: {current_config.object_pool_max_size}")
print(f"Cache size: {current_config.cache_max_size}")
```

#### Update Configuration at Runtime
```python
# Update configuration at runtime
async def update_config(executor, new_config):
    # Validate new configuration
    issues = new_config.validate()
    if issues:
        print(f"Configuration issues: {issues}")
        return False

    # Apply new configuration
    await executor.apply_optimization_config(new_config)

    # Verify application
    current_config = executor.optimization_config
    print(f"Configuration updated: {current_config.to_dict()}")
    return True
```

## Error Handling Issues

### Issue: Circuit Breaker Too Sensitive

**Symptoms:**
- Circuit breaker opens too frequently
- Legitimate operations being blocked
- Recovery takes too long

**Root Causes:**
- Failure threshold too low
- Recovery timeout too long
- Normal errors being treated as failures

**Solutions:**

#### Adjust Circuit Breaker Settings
```python
# Less sensitive circuit breaker
config = OptimizationConfig(
    # Higher failure threshold
    circuit_breaker_failure_threshold=20,  # Increase from default 10

    # Shorter recovery timeout
    circuit_breaker_recovery_timeout_seconds=30,  # Reduce from default 60

    # Enable error handling
    enable_circuit_breaker=True,
    enable_optimized_error_handling=True,
)
```

#### Monitor Circuit Breaker State
```python
# Monitor circuit breaker state
async def monitor_circuit_breaker(executor):
    while True:
        stats = executor.get_optimization_stats()

        if 'circuit_breaker' in stats:
            state = stats['circuit_breaker']['state']
            failure_count = stats['circuit_breaker']['failure_count']

            print(f"Circuit breaker state: {state}")
            print(f"Failure count: {failure_count}")

            if state == 'OPEN':
                print("WARNING: Circuit breaker is open")

        await asyncio.sleep(10)
```

### Issue: Error Recovery Not Working

**Symptoms:**
- Errors not being recovered from
- Repeated failures
- No automatic recovery

**Root Causes:**
- Error handling disabled
- Recovery strategies not configured
- Error patterns not being cached

**Solutions:**

#### Enable Error Handling
```python
# Enable comprehensive error handling
config = OptimizationConfig(
    # Enable error handling
    enable_optimized_error_handling=True,
    enable_circuit_breaker=True,

    # Configure error caching
    error_cache_size=100,  # Increase from default 50

    # Enable automatic recovery
    enable_automatic_optimization=True,
)
```

#### Monitor Error Recovery
```python
# Monitor error recovery effectiveness
async def monitor_error_recovery(executor):
    while True:
        stats = executor.get_optimization_stats()

        if 'error_handling' in stats:
            error_rate = stats['error_handling']['error_rate']
            recovery_success_rate = stats['error_handling']['recovery_success_rate']

            print(f"Error rate: {error_rate:.2%}")
            print(f"Recovery success rate: {recovery_success_rate:.2%}")

            if error_rate > 0.1:  # 10% error rate
                print("WARNING: High error rate detected")

            if recovery_success_rate < 0.5:  # 50% recovery rate
                print("WARNING: Low recovery success rate")

        await asyncio.sleep(10)
```

## Monitoring Issues

### Issue: No Performance Metrics

**Symptoms:**
- No metrics being collected
- Performance monitoring not working
- Missing telemetry data

**Root Causes:**
- Telemetry disabled
- Performance monitoring disabled
- Configuration issues

**Solutions:**

#### Enable Monitoring
```python
# Enable comprehensive monitoring
config = OptimizationConfig(
    # Enable telemetry
    enable_optimized_telemetry=True,
    enable_performance_monitoring=True,

    # Configure telemetry
    telemetry_batch_size=10,
    telemetry_flush_interval_seconds=30.0,
)
```

#### Verify Monitoring Setup
```python
# Verify monitoring is working
async def verify_monitoring(executor):
    # Execute a test step
    result, metrics = await executor.execute_with_monitoring(
        test_step,
        test_data
    )

    print("Monitoring metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Check optimization stats
    stats = executor.get_optimization_stats()
    print("Optimization stats:")
    for category, data in stats.items():
        print(f"  {category}: {data}")
```

### Issue: High Telemetry Overhead

**Symptoms:**
- Performance degradation due to telemetry
- High CPU usage from monitoring
- Slow execution due to metrics collection

**Root Causes:**
- Too frequent telemetry collection
- Large batch sizes
- Excessive monitoring

**Solutions:**

#### Reduce Telemetry Overhead
```python
# Minimize telemetry impact
config = OptimizationConfig(
    # Larger batches, less frequent
    telemetry_batch_size=100,
    telemetry_flush_interval_seconds=60.0,

    # Disable performance monitoring if not needed
    enable_performance_monitoring=False,

    # Keep only essential telemetry
    enable_optimized_telemetry=True,
)
```

#### Use Sampling
```python
# Implement custom sampling
import random

async def sampled_execution(executor, step, data, sample_rate=0.1):
    if random.random() < sample_rate:
        # Only collect metrics for sampled executions
        result, metrics = await executor.execute_with_monitoring(step, data)
        return result, metrics
    else:
        # Regular execution without monitoring
        result = await executor.execute(step, data)
        return result, None
```

## Debugging Tools

### Performance Profiling

```python
# Profile optimization performance
import time
import asyncio

async def profile_optimizations(executor, test_step, test_data, iterations=100):
    times = []

    for i in range(iterations):
        start_time = time.perf_counter()
        result = await executor.execute(test_step, test_data)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(execution_time)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Performance profile:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Standard deviation: {calculate_std(times):.2f}ms")

    return times

def calculate_std(times):
    avg = sum(times) / len(times)
    variance = sum((t - avg) ** 2 for t in times) / len(times)
    return variance ** 0.5
```

### Configuration Debugging

```python
# Debug configuration issues
def debug_configuration(config):
    print("Configuration debug:")
    print(f"  Object pool size: {config.object_pool_max_size}")
    print(f"  Cache size: {config.cache_max_size}")
    print(f"  Telemetry batch size: {config.telemetry_batch_size}")
    print(f"  Max concurrent executions: {config.max_concurrent_executions}")

    # Check for potential issues
    issues = []

    if config.object_pool_max_size > 1000:
        issues.append("Object pool size may be too large")

    if config.cache_max_size > 10000:
        issues.append("Cache size may be too large")

    if config.telemetry_batch_size < 5:
        issues.append("Telemetry batch size may be too small")

    if config.max_concurrent_executions and config.max_concurrent_executions > 50:
        issues.append("Concurrency may be too high")

    if issues:
        print("Potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious issues found")
```

### Memory Debugging

```python
# Debug memory usage
import psutil
import gc

def debug_memory_usage(executor):
    print("Memory debug:")

    # Get system memory info
    memory = psutil.virtual_memory()
    print(f"  System memory usage: {memory.percent:.1f}%")
    print(f"  Available memory: {memory.available / 1024 / 1024:.1f}MB")

    # Get executor stats
    stats = executor.get_optimization_stats()

    if 'memory' in stats:
        memory_stats = stats['memory']
        print(f"  Object pool utilization: {memory_stats.get('object_pool_utilization', 'N/A')}")
        print(f"  Memory usage: {memory_stats.get('usage_mb', 'N/A')}MB")

    # Force garbage collection
    gc.collect()
    print("  Garbage collection completed")

    # Check memory after GC
    memory_after = psutil.virtual_memory()
    print(f"  Memory after GC: {memory_after.percent:.1f}%")
```

### Error Debugging

```python
# Debug error handling
async def debug_error_handling(executor):
    print("Error handling debug:")

    # Get error statistics
    stats = executor.get_optimization_stats()

    if 'error_handling' in stats:
        error_stats = stats['error_handling']
        print(f"  Error rate: {error_stats.get('error_rate', 'N/A')}")
        print(f"  Recovery success rate: {error_stats.get('recovery_success_rate', 'N/A')}")
        print(f"  Cached error patterns: {error_stats.get('cached_patterns', 'N/A')}")

    if 'circuit_breaker' in stats:
        cb_stats = stats['circuit_breaker']
        print(f"  Circuit breaker state: {cb_stats.get('state', 'N/A')}")
        print(f"  Failure count: {cb_stats.get('failure_count', 'N/A')}")
        print(f"  Success count: {cb_stats.get('success_count', 'N/A')}")

    # Test error recovery
    try:
        # Intentionally cause an error
        await executor.execute(lambda: 1/0, None)
    except Exception as e:
        print(f"  Error caught: {type(e).__name__}: {e}")
        print("  Error recovery test completed")
```

## Best Practices for Troubleshooting

### 1. Start Simple
- Begin with minimal configuration
- Enable optimizations one at a time
- Test each change thoroughly

### 2. Monitor Continuously
- Set up monitoring from the start
- Watch for performance regressions
- Track resource usage

### 3. Document Changes
- Keep a log of configuration changes
- Note the impact of each change
- Document successful configurations

### 4. Test with Real Data
- Use production-like data volumes
- Test with realistic concurrency
- Include error scenarios

### 5. Have a Rollback Plan
- Keep baseline configuration
- Know how to disable optimizations
- Have performance benchmarks

## Conclusion

Effective troubleshooting of optimization issues requires systematic approach, proper monitoring, and understanding of the trade-offs involved. By following this guide and using the provided debugging tools, you can identify and resolve optimization-related issues quickly and effectively.

Remember that the goal of optimizations is to improve performance, not to add complexity. If optimizations are causing more problems than they solve, consider using the baseline ExecutorCore instead.
