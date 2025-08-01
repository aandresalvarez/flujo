# ExecutorCore Optimization Configuration Guide

## Overview

This guide provides comprehensive instructions for configuring ExecutorCore optimizations based on workload characteristics, performance requirements, and system constraints. Based on extensive performance testing, this guide emphasizes selective optimization strategies that maximize benefits while minimizing overhead.

## Quick Start

### Default Recommendation: Use Baseline ExecutorCore
For most applications, the baseline ExecutorCore provides optimal performance:

```python
from flujo.application.core.executor_core import ExecutorCore

# Use baseline ExecutorCore (recommended for most workloads)
executor = ExecutorCore()
```

### When to Consider Optimizations
Enable optimizations only when you have:
1. **Identified specific bottlenecks** through profiling
2. **Workload characteristics** that match optimization benefits
3. **Performance testing** that validates improvements

## Optimization Configuration

### Configuration Structure
```python
from flujo.application.core.optimization_config import OptimizationConfig
from flujo.application.core.ultra_executor import UltraExecutor

config = OptimizationConfig(
    # Object pooling
    enable_object_pool=False,
    object_pool_max_size=500,
    object_pool_cleanup_threshold=0.75,
    
    # Context optimization
    enable_context_optimization=False,
    context_cache_size=1024,
    context_cache_ttl=1800,
    
    # Memory optimization
    enable_memory_optimization=False,
    memory_cleanup_interval=300,
    memory_threshold=0.8,
    
    # Concurrency optimization
    enable_concurrency_optimization=False,
    max_concurrent_executions=24,
    concurrency_timeout=30,
    
    # Telemetry optimization
    enable_optimized_telemetry=False,
    telemetry_sampling_rate=0.5,
    telemetry_batch_size=400,
    
    # Additional optimizations
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
    enable_automatic_optimization=False
)

executor = UltraExecutor(config)
```

## Workload-Specific Configurations

### 1. High-Allocation Workloads
**Characteristics**: Frequent object creation/destruction, memory pressure
**Optimizations**: Object pooling, memory management

```python
config = OptimizationConfig(
    # Enable object pooling
    enable_object_pool=True,
    object_pool_max_size=1000,
    object_pool_cleanup_threshold=0.8,
    
    # Enable memory optimization
    enable_memory_optimization=True,
    memory_cleanup_interval=60,
    memory_threshold=0.7,
    
    # Disable other optimizations to minimize overhead
    enable_context_optimization=False,
    enable_concurrency_optimization=False,
    enable_optimized_telemetry=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
    enable_automatic_optimization=False
)
```

### 2. Large Context Workloads  
**Characteristics**: Large execution contexts, frequent context copying
**Optimizations**: Context caching, memory optimization

```python
config = OptimizationConfig(
    # Enable context optimization
    enable_context_optimization=True,
    context_cache_size=2048,
    context_cache_ttl=3600,
    
    # Enable memory optimization
    enable_memory_optimization=True,
    memory_cleanup_interval=120,
    memory_threshold=0.75,
    
    # Disable other optimizations
    enable_object_pool=False,
    enable_concurrency_optimization=False,
    enable_optimized_telemetry=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
    enable_automatic_optimization=False
)
```

### 3. CPU-Intensive Workloads
**Characteristics**: High CPU utilization, parallel processing needs
**Optimizations**: Concurrency optimization, reduced instrumentation

```python
config = OptimizationConfig(
    # Enable concurrency optimization
    enable_concurrency_optimization=True,
    max_concurrent_executions=48,  # 4x CPU cores
    concurrency_timeout=60,
    
    # Minimal telemetry to reduce overhead
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=0.1,  # Low sampling
    telemetry_batch_size=1000,    # Large batches
    
    # Disable other optimizations
    enable_object_pool=False,
    enable_context_optimization=False,
    enable_memory_optimization=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
    enable_automatic_optimization=False
)
```

### 4. High-Frequency Operations
**Characteristics**: Many repeated operations, caching opportunities
**Optimizations**: Cache optimization, telemetry batching

```python
config = OptimizationConfig(
    # Enable cache optimization
    enable_cache_optimization=True,
    cache_size=4096,
    cache_ttl=7200,
    
    # Optimized telemetry for high frequency
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=0.01,  # Very low sampling
    telemetry_batch_size=2000,     # Large batches
    
    # Disable other optimizations
    enable_object_pool=False,
    enable_context_optimization=False,
    enable_memory_optimization=False,
    enable_concurrency_optimization=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_automatic_optimization=False
)
```

### 5. Real-Time/Low-Latency Workloads
**Characteristics**: Strict latency requirements, minimal overhead tolerance
**Optimizations**: Minimal instrumentation, no optimization overhead

```python
# Use baseline ExecutorCore for lowest latency
executor = ExecutorCore()

# Or minimal optimization config if needed
config = OptimizationConfig(
    # Disable all optimizations for minimal overhead
    enable_object_pool=False,
    enable_context_optimization=False,
    enable_memory_optimization=False,
    enable_step_optimization=False,
    enable_algorithm_optimization=False,
    enable_concurrency_optimization=False,
    enable_optimized_telemetry=False,
    enable_performance_monitoring=False,
    enable_optimized_error_handling=False,
    enable_circuit_breaker=False,
    enable_cache_optimization=False,
    enable_automatic_optimization=False
)
```

## Parameter Tuning Guidelines

### Object Pool Configuration
```python
# Small workloads (< 100 operations/sec)
object_pool_max_size=50
object_pool_cleanup_threshold=0.9

# Medium workloads (100-1000 operations/sec)  
object_pool_max_size=500
object_pool_cleanup_threshold=0.75

# Large workloads (> 1000 operations/sec)
object_pool_max_size=2000
object_pool_cleanup_threshold=0.6
```

### Context Cache Configuration
```python
# Small contexts (< 1KB)
context_cache_size=512
context_cache_ttl=900

# Medium contexts (1KB-10KB)
context_cache_size=1024  
context_cache_ttl=1800

# Large contexts (> 10KB)
context_cache_size=2048
context_cache_ttl=3600
```

### Concurrency Configuration
```python
import os

# Conservative (1x CPU cores)
max_concurrent_executions=os.cpu_count()

# Balanced (2x CPU cores) - recommended
max_concurrent_executions=os.cpu_count() * 2

# Aggressive (4x CPU cores) - for I/O bound workloads
max_concurrent_executions=os.cpu_count() * 4
```

### Telemetry Configuration
```python
# Development/debugging
telemetry_sampling_rate=1.0    # 100% sampling
telemetry_batch_size=10        # Small batches

# Production monitoring
telemetry_sampling_rate=0.1    # 10% sampling
telemetry_batch_size=100       # Medium batches

# High-performance production
telemetry_sampling_rate=0.01   # 1% sampling
telemetry_batch_size=1000      # Large batches
```

## System-Specific Tuning

### Memory-Constrained Systems
```python
config = OptimizationConfig(
    # Aggressive cleanup
    object_pool_cleanup_threshold=0.9,
    memory_cleanup_interval=30,
    memory_threshold=0.6,
    
    # Smaller caches
    context_cache_size=256,
    cache_size=512,
    
    # Shorter TTLs
    context_cache_ttl=600,
    cache_ttl=1200,
    
    # Minimal telemetry
    telemetry_sampling_rate=0.01,
    telemetry_batch_size=50
)
```

### High-Memory Systems
```python
config = OptimizationConfig(
    # Larger pools and caches
    object_pool_max_size=5000,
    context_cache_size=8192,
    cache_size=16384,
    
    # Longer TTLs
    context_cache_ttl=7200,
    cache_ttl=14400,
    
    # Less aggressive cleanup
    object_pool_cleanup_threshold=0.5,
    memory_threshold=0.9,
    memory_cleanup_interval=600
)
```

## Performance Validation

### Before Enabling Optimizations
Always validate performance before and after enabling optimizations:

```python
from scripts.performance_validation import run_performance_validation

# Test baseline performance
baseline_results = run_performance_validation(use_optimized=False)

# Test optimized performance  
optimized_results = run_performance_validation(use_optimized=True, config=your_config)

# Compare results
improvement = calculate_improvement(baseline_results, optimized_results)
if improvement < 0:
    print("Warning: Optimizations caused performance regression")
    # Consider disabling optimizations or adjusting parameters
```

### Continuous Performance Monitoring
```python
# Enable performance monitoring for production
config = OptimizationConfig(
    enable_performance_monitoring=True,
    performance_monitoring_interval=300,  # 5 minutes
    performance_alert_threshold=0.2       # 20% degradation
)
```

## Best Practices

### 1. Start Conservative
- Begin with baseline ExecutorCore
- Enable one optimization at a time
- Validate each change with performance testing

### 2. Profile Before Optimizing
```python
import cProfile
import pstats

# Profile your workload
profiler = cProfile.Profile()
profiler.enable()

# Run your workload
your_workload_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

### 3. Monitor in Production
- Use telemetry to track performance metrics
- Set up alerts for performance degradation
- Regularly review optimization effectiveness

### 4. Test with Realistic Workloads
- Use production-like data volumes
- Test with realistic concurrency levels
- Include error scenarios in testing

### 5. Document Configuration Decisions
```python
# Document why specific optimizations are enabled
config = OptimizationConfig(
    # Object pooling enabled due to high allocation rate (profiling showed 80% time in allocation)
    enable_object_pool=True,
    object_pool_max_size=1000,  # Based on peak concurrent operations observed
    
    # Context optimization disabled due to small context sizes (< 1KB average)
    enable_context_optimization=False,
    
    # Telemetry sampling reduced for production performance
    telemetry_sampling_rate=0.05  # 5% sampling sufficient for monitoring
)
```

## Troubleshooting

### Performance Regression After Enabling Optimizations
1. **Disable all optimizations** and test baseline performance
2. **Enable optimizations one by one** to identify the problematic component
3. **Adjust parameters** for the problematic optimization
4. **Consider workload mismatch** - the optimization may not suit your use case

### High Memory Usage
1. **Reduce pool sizes**: `object_pool_max_size`, `context_cache_size`, `cache_size`
2. **Increase cleanup frequency**: Lower `memory_cleanup_interval`
3. **Reduce TTLs**: Lower `context_cache_ttl`, `cache_ttl`
4. **Disable memory-intensive optimizations**: Object pooling, caching

### High CPU Usage
1. **Reduce telemetry sampling**: Lower `telemetry_sampling_rate`
2. **Increase batch sizes**: Higher `telemetry_batch_size`
3. **Disable performance monitoring**: Set `enable_performance_monitoring=False`
4. **Reduce concurrency**: Lower `max_concurrent_executions`

### Inconsistent Performance
1. **Check system resources**: CPU, memory, I/O utilization
2. **Review telemetry data**: Look for patterns in performance metrics
3. **Test with fixed parameters**: Disable automatic optimization
4. **Validate test methodology**: Ensure consistent test conditions

## Configuration Templates

### Development Environment
```python
# Development configuration with full instrumentation
DEV_CONFIG = OptimizationConfig(
    enable_performance_monitoring=True,
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=1.0,
    telemetry_batch_size=1,
    # All other optimizations disabled for debugging
)
```

### Staging Environment  
```python
# Staging configuration matching production but with more monitoring
STAGING_CONFIG = OptimizationConfig(
    enable_performance_monitoring=True,
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=0.1,
    telemetry_batch_size=100,
    # Enable optimizations based on production workload analysis
)
```

### Production Environment
```python
# Production configuration optimized for performance
PRODUCTION_CONFIG = OptimizationConfig(
    enable_performance_monitoring=True,
    enable_optimized_telemetry=True,
    telemetry_sampling_rate=0.01,
    telemetry_batch_size=1000,
    # Selective optimizations based on profiling results
)
```

## Conclusion

ExecutorCore optimization requires careful consideration of workload characteristics, system constraints, and performance requirements. The key to successful optimization is:

1. **Measure first**: Profile and understand your workload
2. **Optimize selectively**: Enable only beneficial optimizations
3. **Validate continuously**: Monitor performance in production
4. **Iterate carefully**: Make incremental changes with validation

Remember: The baseline ExecutorCore is already well-optimized for most use cases. Only add optimization layers when you have clear evidence they will provide benefits for your specific workload.

---

*Guide Version: 1.0*  
*Last Updated: July 31, 2025*  
*Based on: ExecutorCore Optimization Performance Analysis*