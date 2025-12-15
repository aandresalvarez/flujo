# Robustness Tests

This directory contains comprehensive tests to ensure Flujo's robustness across multiple dimensions: performance, memory safety, concurrency, error recovery, configuration validation, and integration stability.

## Overview

Robustness tests go beyond basic functionality to ensure Flujo performs reliably under:
- **Performance pressure** (regressions, scaling)
- **Memory constraints** (leaks, exhaustion)
- **Concurrent access** (thread safety, race conditions)
- **Error conditions** (recovery, degradation)
- **Configuration edge cases** (validation, parsing)
- **Integration scenarios** (cross-component interaction)

## Test Categories

### Performance Regression Tests (`test_performance_regression.py`)

**Purpose**: Detect performance regressions and ensure acceptable performance bounds.

**Key Tests**:
- **Step Execution Performance**: Measures individual step execution time
- **Pipeline Creation Performance**: Tracks pipeline instantiation overhead
- **Context Isolation Performance**: Monitors context isolation speed
- **Serialization Performance**: Measures object serialization speed
- **Memory Overhead**: Tracks memory usage growth
- **Concurrent Execution**: Tests performance under concurrency
- **Caching Effectiveness**: Verifies cache performance improvements

**Failure Indicators**:
- Execution times exceeding baseline thresholds
- Memory growth beyond acceptable limits
- Cache performance below improvement targets

### Memory Leak Detection (`test_memory_leak_detection.py`)

**Purpose**: Detect memory leaks and ensure proper resource cleanup.

**Key Tests**:
- **Repeated Execution Leak Detection**: Monitors object growth over iterations
- **Context Cleanup**: Ensures PipelineContext objects are garbage collected
- **Circular Reference Handling**: Tests cleanup of circular references
- **Large Data Processing**: Monitors memory during big data operations
- **Background Task Cleanup**: Ensures async tasks are properly cleaned up
- **Cache Memory Bounds**: Verifies caching doesn't cause unbounded growth

**Failure Indicators**:
- Uncollected garbage objects
- Memory usage growth beyond thresholds
- Resource handles not properly closed

### Concurrency Safety (`test_concurrency_safety.py`)

**Purpose**: Ensure thread safety and proper concurrent operation handling.

**Key Tests**:
- **Executor Thread Safety**: Tests concurrent step execution
- **Context Isolation Under Concurrency**: Verifies context isolation works across threads
- **Async Task Safety**: Tests async operations under high concurrency
- **Cache Thread Safety**: Ensures caching works safely across threads
- **Shared Resource Safety**: Tests synchronized access to shared resources
- **Async Context Managers**: Verifies async context manager safety
- **Event Loop Isolation**: Tests operation across different event loops

**Failure Indicators**:
- Race conditions causing data corruption
- Deadlocks or thread blocking
- Inconsistent state across concurrent operations

### Error Recovery (`test_error_recovery.py`)

**Purpose**: Ensure graceful error handling and system resilience.

**Key Tests**:
- **Graceful Degradation**: Tests behavior when components fail
- **Network Timeout Recovery**: Handles network-related failures
- **Memory Exhaustion Recovery**: Tests behavior under memory pressure
- **Corrupted Context Recovery**: Handles invalid context data
- **Circular Reference Recovery**: Manages serialization issues
- **Configuration Error Recovery**: Handles invalid configurations
- **Signal Handling**: Tests graceful shutdown during execution
- **Resource Exhaustion**: Handles system resource limits
- **Service Degradation**: Tests behavior under external service issues

**Failure Indicators**:
- System crashes instead of graceful failures
- Resource leaks during error conditions
- Inability to recover from transient failures

### Configuration Robustness (`test_configuration_robustness.py`)

**Purpose**: Ensure configuration handling is robust and well-validated.

**Key Tests**:
- **Configuration Validation**: Rejects invalid configurations
- **YAML/JSON Parsing**: Handles malformed configuration files
- **Environment Variables**: Tests environment-based configuration
- **Missing File Handling**: Graceful handling of missing configs
- **Large File Processing**: Handles large configuration files
- **Special Character Handling**: Processes unicode and special chars
- **Schema Validation**: Ensures configuration structure compliance
- **Inheritance Logic**: Tests configuration merging and overrides
- **Integration Testing**: Full pipeline execution robustness
- **Boundary Conditions**: Tests edge cases and extreme values

**Failure Indicators**:
- Invalid configurations accepted
- Parsing crashes on malformed files
- Configuration inheritance failures

## Running the Tests

### Run All Robustness Tests
```bash
# From project root
pytest tests/robustness/ -v
```

### Run Specific Categories
```bash
# Performance tests only
pytest tests/robustness/test_performance_regression.py -v

# Memory tests only
pytest tests/robustness/test_memory_leak_detection.py -v

# Concurrency tests only
pytest tests/robustness/test_concurrency_safety.py -v

# Error recovery tests only
pytest tests/robustness/test_error_recovery.py -v

# Configuration tests only
pytest tests/robustness/test_configuration_robustness.py -v
```

### Run with Performance Profiling
```bash
# Run performance tests with detailed profiling
pytest tests/robustness/test_performance_regression.py -v -s --durations=10
```

### Run Memory Leak Tests
```bash
# Run memory tests (may be slower)
pytest tests/robustness/test_memory_leak_detection.py -v
```

## Test Configuration

### Baseline Thresholds

Performance tests use configurable thresholds defined in test fixtures:

```python
baseline_thresholds = {
    "step_execution": 50.0,      # Max ms per step
    "pipeline_creation": 10.0,   # Max ms for pipeline creation
    "context_isolation": 5.0,    # Max ms for context isolation
    "serialization": 20.0,       # Max ms for serialization
    "memory_overhead": 10.0,     # Max % memory growth
}
```

### Environment Variables

Some tests respect environment variables for configuration:

- `FLUJO_ROBUSTNESS_TEST_ITERATIONS`: Number of iterations for stress tests (default: 100)
- `FLUJO_ROBUSTNESS_MEMORY_THRESHOLD`: Memory growth threshold percentage (default: 10)
- `FLUJO_ROBUSTNESS_TIMEOUT`: Test timeout in seconds (default: 300)

## Test Failure Analysis

### Common Failure Patterns

#### Performance Regressions
```
FAILED test_performance_regression.py::TestPerformanceRegression::test_step_execution_performance
> Execution time 75.0ms exceeded threshold 50.0ms
```
**Action**: Investigate algorithm changes, caching issues, or resource contention.

#### Memory Leaks
```
FAILED test_memory_leak_detection.py::TestMemoryLeakDetection::test_no_executor_memory_leak_on_repeated_execution
> Memory leak detected: 150 new objects after 100 iterations
```
**Action**: Check for object retention, circular references, or improper cleanup.

#### Concurrency Issues
```
FAILED test_concurrency_safety.py::TestConcurrencySafety::test_executor_thread_safety
> Thread execution errors: [Exception('Data corruption detected')]
```
**Action**: Review shared state access, add proper synchronization.

#### Configuration Issues
```
FAILED test_configuration_robustness.py::TestConfigurationValidation::test_invalid_executor_configuration_rejection
> Configuration validation failed to reject invalid cache_size
```
**Action**: Strengthen configuration validation logic.

## Integration with CI/CD

### GitHub Actions Example
```yaml
# .github/workflows/robustness.yml
name: Robustness Tests
on: [push, pull_request]

jobs:
  robustness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install dependencies
        run: |
          pip install -e .[test]

      - name: Run robustness tests
        run: |
          pytest tests/robustness/ -v --durations=20 --tb=short

      - name: Performance regression check
        run: |
          pytest tests/robustness/test_performance_regression.py::TestPerformanceRegression::test_step_execution_performance -v
```

### Pre-commit Integration
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: robustness-tests
      name: Critical Robustness Tests
      entry: pytest tests/robustness/test_error_recovery.py tests/robustness/test_memory_leak_detection.py --tb=line
      language: system
      pass_filenames: false
```

## Monitoring and Metrics

### Performance Baselines

Track these metrics over time:
- Average step execution time
- Memory usage per operation
- Concurrent operation throughput
- Error recovery success rate
- Configuration parsing time

### Automated Reporting

Consider integrating with:
- **Performance monitoring**: Track execution times in CI
- **Memory profiling**: Use `memory_profiler` for detailed analysis
- **Coverage reporting**: Ensure robustness tests maintain coverage
- **Regression detection**: Alert on performance degradation

## Extending the Tests

### Adding New Robustness Tests

1. **Choose appropriate file** based on test category
2. **Follow naming convention**: `test_descriptive_name`
3. **Include failure thresholds** where applicable
4. **Add documentation** explaining what the test verifies
5. **Update this README** with new test descriptions

### Example: Adding Network Resilience Test
```python
def test_network_resilience_under_packet_loss(self):
    """Test behavior under simulated network packet loss."""
    # Implementation here
    pass
```

## Troubleshooting

### Test Timeouts
- Increase timeout values in environment variables
- Check for infinite loops in test code
- Verify external dependencies are responding

### Memory Issues
- Run tests individually to isolate memory hogs
- Use `memory_profiler` to identify memory-intensive tests
- Check for object retention in test fixtures

### Concurrency Flakiness
- Run tests multiple times to identify race conditions
- Add synchronization primitives if needed
- Review shared state access patterns

## Related Documentation

- [`FLUJO_TEAM_GUIDE.md`](../../FLUJO_TEAM_GUIDE.md) - Architectural standards
- [`docs/development/type_safety.md`](../../docs/development/type_safety.md) - Type safety patterns
- [`tests/architecture/`](../architecture/) - Architecture compliance tests
- [`tests/test_types/`](../test_types/) - Type-safe test fixtures

## Contributing

When adding robustness tests:

1. **Categorize properly** - Choose the right test file based on focus area
2. **Include baselines** - Define acceptable thresholds for performance/memory tests
3. **Document failures** - Explain what each test failure indicates
4. **Test isolation** - Ensure tests don't interfere with each other
5. **Performance awareness** - Keep test execution time reasonable

These robustness tests ensure Flujo remains reliable and performant under production conditions and edge cases.
