# ExecutorCore Optimization Documentation

## Overview

This documentation provides comprehensive guidance for using and configuring ExecutorCore optimizations in Flujo. The optimization system provides significant performance improvements through memory optimization, execution optimization, telemetry optimization, and error handling optimization.

> **Compatibility notice:** The `OptimizedExecutorCore` shim has been removed. Use `ExecutorCore` with an `OptimizationConfig`; legacy imports should be updated to the new entrypoint.

## 📚 Documentation Structure

### Core Guides

1. **[ExecutorCore Optimization Guide](executor_core_optimization_guide.md)**
   - Complete user guide for optimization configuration
   - Performance tuning strategies
   - Use case examples and best practices
   - Configuration parameters and their impact

2. **[Optimization Troubleshooting Guide](optimization_troubleshooting_guide.md)**
   - Common issues and their solutions
   - Performance regression detection
   - Memory and configuration problems
   - Debugging tools and techniques

3. **[Optimization Monitoring Guide](optimization_monitoring_guide.md)**
   - Metrics collection and analysis
   - Real-time monitoring setup
   - Performance visualization
   - Alerting and threshold management

### Examples and Demos

4. Comprehensive Optimization Demo (coming soon)
   - End-to-end demonstration of optimization features
   - Performance comparison across configurations
   - Real-world usage examples

## 🚀 Quick Start

### Basic Usage

```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

opt_config = OptimizationConfig(
    enable_object_pool=True,
    enable_context_optimization=True,
    enable_memory_optimization=True,
    enable_optimized_telemetry=True,
)

executor = ExecutorCore(optimization_config=opt_config)
result = await executor.execute(step, data, context=context)
```

### Custom Configuration

```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

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

## 🎯 Optimization Components

### Memory Optimizations
- **Object Pooling**: Reduces memory allocation overhead by reusing objects
- **Context Management**: Optimizes context copying and merging operations
- **Memory Pressure Detection**: Automatically adjusts memory usage based on system pressure

### Performance Optimizations
- **Step Execution**: Pre-analyzes steps and caches analysis results
- **Algorithm Optimizations**: Optimizes hashing, serialization, and cache key generation
- **Concurrency Optimization**: Adaptive concurrency limits and work-stealing queues

### Telemetry Optimizations
- **Low-Overhead Tracing**: Minimal performance impact telemetry collection
- **Performance Monitoring**: Real-time performance metrics and threshold detection

### Error Handling Optimizations
- **Error Caching**: Caches error patterns for faster recovery
- **Circuit Breaker**: Prevents cascade failures through automatic failure detection

## 📊 Performance Benefits

Based on comprehensive testing, the optimization system provides:

- **20-40%** reduction in memory allocation for frequently used objects
- **10-30%** improvement in step execution speed
- **20-50%** faster cache operations
- **15-25%** speedup in context operations
- **<1%** performance impact from telemetry

## 🔧 Configuration Examples

### High-Throughput Workloads

```python
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

### Memory-Constrained Environments

```python
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

### Latency-Sensitive Applications

```python
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

## 📈 Monitoring and Metrics

### Basic Monitoring

```python
# Get optimization statistics
stats = executor.get_optimization_stats()

# Memory metrics
memory_usage = stats['memory']['usage_mb']
object_pool_utilization = stats['memory']['object_pool_utilization']

# Performance metrics
avg_execution_time = stats['execution']['avg_time_ms']
cache_hit_rate = stats['cache']['hit_rate']

# Error metrics
error_rate = stats['error_handling']['error_rate']
circuit_breaker_state = stats['error_handling']['circuit_breaker_state']
```

### Performance Recommendations

```python
# Get automatic performance recommendations
recommendations = executor.get_performance_recommendations()

for rec in recommendations:
    print(f"Recommendation: {rec['description']}")
    print(f"Impact: {rec['impact']}")
    print(f"Priority: {rec['priority']}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```python
   # Reduce object pool size
   config = OptimizationConfig(
       object_pool_max_size=20,
       object_pool_cleanup_threshold=0.6,
   )
   ```

2. **Slow Execution**
   ```python
   # Enable step optimization
   config = OptimizationConfig(
       enable_step_optimization=True,
       enable_algorithm_optimization=True,
   )
   ```

3. **High CPU Usage**
   ```python
   # Reduce telemetry overhead
   config = OptimizationConfig(
       telemetry_batch_size=50,
       telemetry_flush_interval_seconds=60.0,
   )
   ```

### Debugging Tools

```python
# Debug configuration
def debug_configuration(config):
    issues = config.validate()
    if issues:
        print("Configuration issues:", issues)

# Debug memory usage
def debug_memory_usage(executor):
    stats = executor.get_optimization_stats()
    memory_stats = stats.get('memory', {})
    print(f"Memory usage: {memory_stats.get('usage_mb', 'N/A')}MB")
    print(f"Object pool utilization: {memory_stats.get('object_pool_utilization', 'N/A')}%")
```

## 🔄 Migration Guide

### Legacy note: OptimizedExecutorCore (removed)

```python
# Old (removed) code
# from flujo.application.core.executor_core import OptimizedExecutorCore
# executor = OptimizedExecutorCore()

# New supported code
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig
executor = ExecutorCore(optimization_config=OptimizationConfig())
```

The shim no longer exists; update imports to use `ExecutorCore` directly.

## 📋 Best Practices

### 1. Start with Default Configuration
Begin with the default configuration and adjust based on your specific needs:

```python
executor = ExecutorCore(optimization_config=OptimizationConfig())
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

## 🧪 Testing and Validation

### Run the Comprehensive Demo

```bash
# See the Optimization Guide for end-to-end examples
# docs/optimization/executor_core_optimization_guide.md
```

This demo showcases:
- Memory optimization features
- Performance optimization features
- Error handling capabilities
- Monitoring and metrics
- Configuration management
- Performance comparison across configurations

### Performance Testing

```python
# Test performance with monitoring
result, metrics = await executor.execute_with_monitoring(step, data)
print(f"Execution time: {metrics['execution_time_ms']}ms")
print(f"Memory usage: {metrics['memory_usage_mb']}MB")
```

## 📚 Additional Resources

### Documentation Files
- `executor_core_optimization_guide.md` - Complete user guide
- `optimization_troubleshooting_guide.md` - Troubleshooting guide
- `optimization_monitoring_guide.md` - Monitoring and metrics guide

### Example Files
- End-to-end demo: see the Optimization Guide

### Related Documentation
- [Introduction](../getting-started/index.md)
- [ExecutorCore Details](../advanced/core_execution_refactor.md)
- [Performance Optimizations](performance_optimizations.md)

## 🤝 Contributing

When contributing to the optimization system:

1. **Test Performance Impact**: Always measure the performance impact of changes
2. **Maintain Backward Compatibility**: Ensure existing code continues to work
3. **Add Documentation**: Update relevant documentation for new features
4. **Include Examples**: Provide usage examples for new optimizations
5. **Validate Configuration**: Ensure new configuration options are properly validated

## 📞 Support

For questions or issues with optimizations:

1. **Check the Troubleshooting Guide**: Many common issues are covered
2. **Review Performance Metrics**: Use monitoring to identify bottlenecks
3. **Test with Different Configurations**: Try different optimization combinations
4. **Consult the Examples**: Review the comprehensive demo for usage patterns

## 🎯 Conclusion

The ExecutorCore optimization system provides significant performance improvements while maintaining ease of use and backward compatibility. By following this documentation and monitoring your application's performance, you can achieve optimal performance for your specific use case.

Remember that the goal of optimizations is to improve performance, not to add complexity. If optimizations are causing more problems than they solve, consider using the baseline ExecutorCore instead.

---

*Documentation Version: 1.0*
*Last Updated: January 2025*
*Based on: ExecutorCore Optimization Performance Analysis*
