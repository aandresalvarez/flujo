# Test Performance Guide

This guide explains the test performance optimizations implemented in the Flujo project and how to use the new test commands effectively.

## Overview

The test suite has been optimized for better performance through:

1. **Parallel Test Execution** - Using pytest-xdist for concurrent test runs
2. **Test Categorization** - Separating fast, slow, and serial tests
3. **Performance Monitoring** - Tools to analyze test performance
4. **Optimized Test Commands** - New Makefile targets for different testing scenarios

## Performance Improvements

### Before Optimization
- **Total test time**: ~5 minutes 19 seconds
- **All tests run serially**
- **No test categorization**

### After Optimization
- **Fast tests**: ~25 seconds (92% improvement)
- **Slow tests**: ~3 minutes 43 seconds (run separately)
- **Parallel execution** with 8 workers
- **Smart test categorization**

## Test Categories

### Fast Tests
- **Execution**: Parallel with 8 workers
- **Duration**: ~25 seconds
- **Includes**: Unit tests, integration tests (excluding slow ones)
- **Command**: `make test-fast`

### Slow Tests (CI-Optimized)
- **Execution**: Serial (to avoid resource conflicts)
- **Duration**: ~46 seconds (optimized for mass CI)
- **Includes**: Performance tests, benchmarks, serial tests (excludes ultra-slow)
- **Command**: `make test-slow`

### Ultra-Slow Tests (Mass CI Problematic)
- **Execution**: Serial, excluded from regular CI
- **Duration**: >30 seconds per test
- **Includes**: Stress tests, sustained load tests
- **Command**: `make test-ultra-slow`
- **Marked with**: `@pytest.mark.ultra_slow`

### Serial Tests
- **Execution**: Always serial
- **Reason**: SQLite concurrency, resource sharing, etc.
- **Marked with**: `@pytest.mark.serial`

## New Test Commands

### Quick Development Feedback
```bash
# Run fast tests in parallel (recommended for development)
make test-fast

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration
```

### Comprehensive Testing
```bash
# Run all tests (original behavior)
make test

# Run all tests in parallel (excluding serial tests)
make test-parallel

# Run slow tests separately (excludes ultra-slow)
make test-slow

# Run ultra-slow tests (problematic for mass CI)
make test-ultra-slow
```

### Coverage Testing
```bash
# Run all tests with coverage
make testcov

# Run fast tests with coverage in parallel
make testcov-fast
```

### Performance Analysis
```bash
# Analyze test performance and identify slow tests
make test-perf
```

### Specific Test Categories
```bash
# Run benchmark tests only
make test-bench

# Run end-to-end tests only
make test-e2e
```

## Test Markers

The following pytest markers are used for test categorization:

- `@pytest.mark.fast` - Fast tests that can run in parallel
- `@pytest.mark.slow` - Slow tests that should run separately
- `@pytest.mark.serial` - Tests that must run serially
- `@pytest.mark.benchmark` - Performance benchmark tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.asyncio` - Async tests (handled automatically)

## Performance Monitoring

### Running Performance Analysis
```bash
make test-perf
```

This command:
1. Runs all tests with timing
2. Analyzes test performance
3. Identifies slow tests
4. Provides recommendations
5. Saves detailed report to `test_performance_report.json`

### Performance Report Output
```
📊 Test Performance Report
==================================================
Total Tests: 10
Total Time: 214.33s
Average Time: 21.433s
Slow Threshold: 50.430s

📁 Test Categories:
  Unit: 8 tests, 206.69s
  Benchmark: 2 tests, 7.64s

🐌 Top 10 Slowest Tests:
   1. tests/unit/test_persistence_performance.py::test_lens_list_performance (50.430s)
   2. tests/unit/test_persistence_performance.py::test_lens_show_performance (49.590s)
   ...

💡 Performance Recommendations:
  • Use 'make test-fast' for faster feedback during development
  • Use 'make test-parallel' for parallel execution
  • Use 'make test-unit' for quick unit test feedback
```

## CI/CD Integration

The CI pipeline has been updated to use the new test categorization:

```yaml
# Run non-SQLite tests with parallel mode for better performance
uv run coverage run --source=flujo --parallel-mode -m pytest tests/ -k "not sqlite and not test_sqlite"

# Run SQLite tests separately without parallel mode to avoid bus errors
uv run coverage run --source=flujo --append -m pytest tests/ -k "sqlite or test_sqlite"
```

## Best Practices

### For Developers
1. **Use `make test-fast`** for quick feedback during development
2. **Use `make test-unit`** for unit test changes
3. **Use `make test`** for comprehensive testing before commits
4. **Use `make test-perf`** to identify performance issues

### For CI/CD
1. **Fast tests run in parallel** for quick feedback
2. **Slow tests run separately** to avoid resource conflicts
3. **Coverage is collected** from both fast and slow test runs
4. **Performance monitoring** can be added to detect regressions

### For Performance Testing
1. **Benchmark tests** are marked as slow and run separately
2. **Performance thresholds** are configurable via environment variables
3. **Memory profiling** is available for resource-intensive tests
4. **Concurrent testing** is disabled for tests that require isolation

## Configuration

### Pytest Configuration
The `pyproject.toml` includes optimized pytest settings:

```toml
[tool.pytest.ini_options]
addopts = [
  "--strict-markers",
  "--strict-config",
  "--tb=short",
]
markers = [
  "slow: marks tests that are slow and should be run separately",
  "fast: marks tests that are fast and can be run in parallel",
  "serial: marks tests that must run serially to avoid concurrency issues",
]
```

### Environment Variables
- `FLUJO_OVERHEAD_LIMIT` - Performance overhead threshold (default: 15.0)
- `FLUJO_CLI_PERF_THRESHOLD` - CLI performance threshold (default: 0.2)
- `FLUJO_CV_THRESHOLD` - Coverage threshold (default: 1.0)

## Troubleshooting

### Common Issues

1. **Tests failing in parallel**
   - Check if tests are properly marked as `serial`
   - Ensure tests don't share resources (files, databases)

2. **Slow test execution**
   - Use `make test-perf` to identify slow tests
   - Consider marking very slow tests with `@pytest.mark.slow`

3. **Memory issues**
   - Some tests may require more memory
   - Consider running memory-intensive tests separately

4. **CI failures**
   - Check if tests are compatible with CI environment
   - Ensure proper test categorization for CI pipeline

### Performance Tips

1. **Use appropriate test commands**:
   - `make test-fast` for development
   - `make test` for comprehensive testing
   - `make test-slow` for performance validation

2. **Monitor test performance**:
   - Run `make test-perf` regularly
   - Check for new slow tests
   - Optimize or categorize slow tests appropriately

3. **Optimize test fixtures**:
   - Use `@pytest.fixture(scope="session")` for expensive setup
   - Consider lazy loading for heavy resources
   - Use appropriate fixture scopes

## Future Improvements

1. **Test Sharding** - Split tests across multiple CI workers
2. **Caching** - Cache test results for faster re-runs
3. **Incremental Testing** - Only run tests affected by changes
4. **Performance Regression Detection** - Automated performance monitoring
5. **Test Parallelization** - Further optimize parallel execution strategies

## Contributing

When adding new tests:

1. **Categorize appropriately**:
   - Use `@pytest.mark.slow` for performance tests
   - Use `@pytest.mark.serial` for tests requiring isolation
   - Use `@pytest.mark.fast` for quick unit tests

2. **Consider performance impact**:
   - Keep unit tests fast (< 1 second)
   - Mark integration tests appropriately
   - Use appropriate test fixtures

3. **Update documentation**:
   - Document new test categories
   - Update performance guidelines
   - Add troubleshooting tips if needed
