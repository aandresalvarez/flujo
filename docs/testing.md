# Testing Guide: Robust Test Management Implementation

This document provides comprehensive details about Flujo's robust test management implementation, including the new Makefile targets, pytest optimizations, and best practices for developers.

## Overview

Flujo's test suite has been completely optimized for better performance, reliability, and developer experience. The implementation includes:

- **Parallel Test Execution** with pytest-xdist
- **Smart Test Categorization** (fast, slow, serial)
- **Comprehensive Test Commands** for different scenarios
- **Performance Monitoring** and analysis tools
- **Hang Detection** and timeout management
- **Advanced Debugging** capabilities

## Performance Improvements

- Parallel execution by default (`-n auto` with load balancing).
- Clear categorization: fast vs slow/serial/benchmark to avoid conflicts.
- Proactive hang protection via timeouts and fault handler.
- Targeted tooling for profiling, sharding, randomization, and reruns.

## Test Categories

### Fast Tests
- **Execution**: Parallel with auto‑detected workers (`-n auto`)
- **Includes**: Unit and integration tests (excluding slow/serial/benchmark)
- **Command**: `make test-fast`
- **Use Case**: Development feedback, quick validation

### Slow Tests
- **Execution**: Serial (to avoid resource conflicts)
- **Includes**: Performance tests, benchmarks, serial tests
- **Command**: `make test-slow`
- **Use Case**: Performance validation, comprehensive testing

#### What belongs in slow tests?
- Benchmarks and performance measurements (`@pytest.mark.benchmark` + `@pytest.mark.slow`).
- Human-in-the-loop (HITL) and stateful resume tests (SQLite-backed) — mark `@pytest.mark.slow` and `@pytest.mark.serial`.
- Trace replay/persistence integration — mark `@pytest.mark.slow`.

### Serial Tests
- **Execution**: Always serial
- **Reason**: SQLite concurrency, resource sharing, etc.
- **Marked with**: `@pytest.mark.serial`
- **Use Case**: Tests requiring isolation

## Core Test Commands

### Quick Development Feedback
```bash
# Run fast tests in parallel (recommended for development)
make test-fast

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run fast tests serially with hang guard (debug parallel issues)
make test-fast-serial
```

### Comprehensive Testing
```bash
# Run all tests (original behavior)
make test

# Run all tests in parallel (excluding serial tests)
make test-parallel

# Run slow tests separately
make test-slow
```

### Coverage Testing
```bash
# Run all tests with coverage
make testcov

# Run fast tests with coverage in parallel
make testcov-fast
```

## Advanced Test Commands

### Performance Analysis
```bash
# Analyze test performance and identify slow tests
make test-perf

# Show the slowest offenders (fast subset focus)
make test-top-slowest

# Quick async plugin sanity check
make test-quick-check
```

### Hang Detection and Prevention
```bash
# Detect and handle hanging tests
make test-hang-guard

# Strict timeout enforcement
make test-timeout-strict
```

### Test Isolation and Debugging
```bash
# Run tests in isolated processes
make test-forked

# Force serial execution for debugging
make test-serial

# Run only tests affected by changes
make test-changed
```

### Advanced Execution Strategies
```bash
# Deterministic test sharding for CI (0-indexed input)
# The Makefile converts 0-indexed SHARD_INDEX to the 1-indexed value
# required by pytest-split's --group flag under the hood.
make test-shard SHARD_INDEX=0 SHARD_COUNT=4

# Randomize test execution order
make test-random-order

# Loop failing tests until they pass
make test-loop
```

### Quality Assurance
```bash
# Find unused test fixtures
make test-deadfixtures

# Profile overall test runtime (writes `test_profile.pstats`)
make test-profile

# Auto-rerun flaky tests
make test-flake-harden
```

### Collection and Analysis
```bash
# Fast test discovery without execution
make test-collection-only

# Comprehensive performance analysis
make test-analyze-performance

# Quick analysis without execution
make test-slow-analysis

# Executor hot-path profiling (cProfile)
uv run python scripts/profile_executor_core_hotpath.py --iterations 50 --warmup 10 --optimized --no-cache
uv run python scripts/compare_profiles.py --a prof/executor_core_execute_no_cache.prof --b prof/executor_core_execute_no_cache_fastpath_optimized.prof --top 30
```

## Pytest Configuration

### Global Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--tb=short",
    "-ra",
    "--durations=25",
    "--durations-min=0.5",
    "-q",
]

markers = [
    "e2e: marks tests as end-to-end (requires network access or VCR cassettes)",
    "benchmark: marks tests as benchmarks (requires pytest-benchmark)",
    "asyncio: marks tests as async (handled by pytest-asyncio)",
    "serial: marks tests that must run serially to avoid concurrency issues",
    "slow: marks tests that are slow and should be run separately",
    "fast: marks tests that are fast and can be run in parallel",
    "integration: marks integration tests for subsystems like the Architect",
    "no_collect: marks classes that should not be collected as test classes",
    "stress: marks stress/resource tests",
    "memory: marks memory-focused tests",
    "timeout: marks tests with custom timeout (override global)",
    "hypothesis: marks property-based tests (requires hypothesis)",
]

filterwarnings = [
    "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited:RuntimeWarning",
    "ignore::pytest.PytestUnraisableExceptionWarning",
]
```

### Environment Variables
No global environment variables are required. Optionally, you can use `PYTEST_ADDOPTS` locally to tweak runs:

```bash
# Narrow the run (example): only a single test by keyword
PYTEST_ADDOPTS='-k test_base_validator_initialization' make test-fast

# Increase verbosity and enable live logs
PYTEST_ADDOPTS='-v -o log_cli=true --log-cli-level=DEBUG' make test-fast
```

## Test Markers and Categorization

### Automatic Markers
```python
import pytest

@pytest.mark.fast
async def test_fast_unit():
    """Fast test that can run in parallel."""
    pass

@pytest.mark.slow
async def test_performance():
    """Slow test that should run separately."""
    pass

@pytest.mark.serial
async def test_sqlite_operation():
    """Test that must run serially."""
    pass

@pytest.mark.benchmark
async def test_benchmark():
    """Performance benchmark test."""
    pass

@pytest.mark.e2e
async def test_end_to_end():
    """End-to-end integration test."""
    pass
```

### Custom Markers
```python
@pytest.mark.stress
async def test_stress():
    """Stress test with high load."""
    pass

@pytest.mark.memory
async def test_memory_intensive():
    """Memory-intensive test."""
    pass

@pytest.mark.timeout(120)
async def test_long_operation():
    """Test with specific timeout."""
    pass
```

### Marking Guidance (Required)
- Prefer module-level `pytestmark` for uniform categorization in benchmark/HITL modules.
- Ensure any test that interacts with interactive steps or persistent state backends is marked `slow` (and `serial` when DB contention is possible).
- Avoid adding kexpr-based exclusions in Makefiles for specific tests; use markers so `make test-fast` selection remains stable.

## Parallel Execution Strategy

### Worker Configuration
```bash
# Auto-detect optimal worker count
-n auto

# Fixed worker count for CI
-n 4

# Load balancing strategy
--dist=loadfile
```

### Load Balancing
- **`--dist=loadfile`**: Distributes tests by file to minimize resource conflicts
- **`--dist=loadscope`**: Distributes tests by scope (session, module, class, function)
- **`--dist=loadgroup`**: Distributes tests by group markers

### Resource Management
```bash
# Fork processes for isolation
--forked

# Timeout protection
--timeout=90
```

## Hang Detection and Prevention

### Timeout Configuration
```bash
# Global timeout (90 seconds)
--timeout=90

# Per-test timeout
--timeout=30

# Fault handler timeout
--faulthandler-timeout=60
```

### Hang Detection Strategies
1. **Process-level timeouts**: Kill hanging processes
2. **Fault handler timeouts**: Detect infinite loops
3. **Resource monitoring**: Track memory and CPU usage
4. **Automatic cleanup**: Clean up resources after timeouts

## Performance Monitoring

### Test Performance Analysis
```bash
make test-perf
```

This command provides:
- **Test timing analysis**: Identify slow tests
- **Performance recommendations**: Optimize test execution
- **Detailed reporting**: JSON report for CI integration
- **Threshold monitoring**: Alert on performance regressions

### Performance Metrics
- **Execution time**: Per-test and aggregate timing
- **Memory usage**: Track memory consumption
- **CPU utilization**: Monitor resource usage
- **Parallel efficiency**: Measure parallel execution benefits

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.13, 3.14]
        test-type: [fast, slow, coverage]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run fast tests
      if: matrix.test-type == 'fast'
      run: make test-fast

    - name: Run slow tests
      if: matrix.test-type == 'slow'
      run: make test-slow

    - name: Run coverage tests
      if: matrix.test-type == 'coverage'
      run: make testcov
```

### Parallel CI Execution
```yaml
# Run tests in parallel across multiple workers
strategy:
  matrix:
    worker: [1, 2, 3, 4]
    include:
      - worker: 1
        test-command: "make test-shard SHARD_INDEX=0 SHARD_COUNT=4"
      - worker: 2
        test-command: "make test-shard SHARD_INDEX=1 SHARD_COUNT=4"
      - worker: 3
        test-command: "make test-shard SHARD_INDEX=2 SHARD_COUNT=4"
      - worker: 4
        test-command: "make test-shard SHARD_INDEX=3 SHARD_COUNT=4"
```

## Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. Tests Failing in Parallel
```bash
# Check for resource conflicts
make test-serial

# Identify problematic tests
make test-perf

# Run with isolation
make test-forked
```

#### 2. Hanging Tests
```bash
# Use hang guard
make test-hang-guard

# Strict timeout enforcement
make test-timeout-strict

# Process isolation
make test-forked
```

#### 3. Memory Issues
```bash
# Run memory-intensive tests separately
pytest -m memory

# Monitor resource usage
make test-analyze-performance
```

### Debug Commands
```bash
# Enable verbose output
make test-fast-verbose

# Run with debug logging
PYTEST_ADDOPTS='-o log_cli=true --log-cli-level=DEBUG' make test-fast

# Collect tests without running
make test-collection-only
```

## CI Stability & Test Isolation

This section documents patterns to prevent "passes in PR, fails in main" scenarios.

### Test Isolation Fixtures

The following autouse fixtures in `tests/conftest.py` ensure proper test isolation:

1. **`_reset_config_cache`**: Clears config manager caches after each test
2. **`_clear_state_uri_env`**: Prevents `FLUJO_STATE_URI` leakage
3. **`_clear_project_root_env`**: Prevents `FLUJO_PROJECT_ROOT` leakage
4. **`_reset_validation_overrides`**: Resets validation rule caches
5. **`_reset_skill_registry_defaults`**: Ensures builtins are registered

### Config Mocking Pattern

**❌ Bad Pattern** (relies on config not being cached):
```python
m.setenv("FLUJO_CONFIG_PATH", str(config_path))
# Config may already be cached from a previous test!
```

**✅ Good Pattern** (mocks at point of use):
```python
from flujo.infra.config import ProviderPricing

def mock_get_cost_config():
    class MockCostConfig:
        strict = True
        providers = {"openai": {"gpt-4o": ProviderPricing(...)}}
    return MockCostConfig()

m.setattr("flujo.infra.config.get_cost_config", mock_get_cost_config)
```

### Performance Test Guidelines

**Avoid tight timing thresholds** - CI environments have high variance due to VM scheduling.

**❌ Flaky Pattern** (tight thresholds):
```python
assert execution_time < 0.01  # 10ms - will fail randomly in CI
assert ratio < 2.0  # Too tight for micro-timing
```

**✅ Stable Pattern** (generous thresholds):
```python
# Use the helper from conftest.py
from tests.conftest import assert_no_major_regression

assert_no_major_regression(
    actual_time=execution_time,
    baseline_time=baseline,
    operation_name="HITL step execution",
    max_ratio=10.0,      # 10x tolerance for CI variance
    absolute_max=30.0,   # Sanity check only
)

# Or for micro-benchmarks, log-only pattern:
print(f"Execution time: {execution_time * 1000:.2f}ms")
assert execution_time < 1.0, "Major regression detected"  # 1s sanity check
```

### Threshold Guidelines

| Operation Type | Recommended Threshold |
|----------------|----------------------|
| Micro-ops (< 10ms expected) | 1s absolute max, log metrics |
| Relative comparisons | 10x ratio max |
| Large operations | 30-60s absolute max |
| Variance checks | 5x max variance |

### StateMachine Test Isolation

Tests that check `context.current_state` are prone to race conditions under xdist.

**Required markers for StateMachine integration tests:**
```python
@pytest.mark.serial  # Race conditions under xdist
@pytest.mark.slow    # If using HITL or SQLite
async def test_state_machine_transitions():
    ...
```

**Files that typically need these markers:**
- Tests checking `current_state` on context
- HITL tests with pause/resume
- Tests using SQLite backends

### Root Causes Reference

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| Config not picked up | Config manager cached | Mock `get_cost_config()` directly |
| `current_state` is None | Race condition in xdist | Mark as `@pytest.mark.serial` |
| 5.16x > 5.0x threshold | Micro-timing variance | Use 10x+ thresholds or log-only |
| HITL test fails randomly | State pollution | Mark as `@pytest.mark.slow` + `@pytest.mark.serial` |

---

## Best Practices

### For Developers
1. **Use appropriate test commands**:
   - `make test-fast` for development feedback
   - `make test-unit` for unit test changes
   - `make test` for comprehensive testing

2. **Mark tests appropriately**:
   - Use `@pytest.mark.slow` for performance tests
   - Use `@pytest.mark.serial` for tests requiring isolation
   - Use `@pytest.mark.fast` for quick unit tests

3. **Optimize test performance**:
   - Keep unit tests fast (< 1 second)
   - Use appropriate test fixtures
   - Avoid unnecessary I/O in fast tests

### For CI/CD
1. **Parallel execution**: Use parallel workers for fast tests
2. **Test categorization**: Run slow tests separately
3. **Performance monitoring**: Track test execution times
4. **Resource management**: Monitor memory and CPU usage

### For Performance Testing
1. **Benchmark isolation**: Run benchmarks separately
2. **Resource monitoring**: Track memory and CPU usage
3. **Threshold management**: Set performance thresholds
4. **Regression detection**: Monitor for performance regressions

## Advanced Features

### Test Sharding
```bash
# Deterministic sharding for CI via Makefile
make test-shard SHARD_INDEX=0 SHARD_COUNT=4

# Raw pytest-split example (without Makefile)
# Note: --group is 1-indexed when calling pytest directly
pytest tests/ --splits 8 --group 2 -p pytest_split
```

### Test Selection
```bash
# Run tests matching patterns
pytest -k "test_agent"

# Exclude specific tests
pytest -k "not slow"

# Run tests by markers
pytest -m "fast and not benchmark"
```

### Coverage Optimization
```bash
# Parallel coverage collection
make testcov-fast

# Coverage with specific tests
pytest --cov=flujo tests/unit/

# Coverage reporting
make testcov-html
```

## Monitoring and Metrics

### Performance Tracking
```bash
# Generate performance report
make test-perf

# Monitor test execution times
make test-analyze-performance

# Profile overall test runtime (writes `test_profile.pstats`)
make test-profile
```

### Quality Metrics
## Future Enhancements
Planned areas include incremental test selection improvements and CI performance regression tracking.

## Troubleshooting Guide

### Performance Issues
1. **Slow test execution**: Use `make test-perf` to identify bottlenecks
2. **Memory issues**: Run memory-intensive tests separately
3. **Parallel conflicts**: Use `make test-serial` for problematic tests

### Reliability Issues
1. **Flaky tests**: Use `make test-flake-harden` to identify and fix
2. **Hanging tests**: Use `make test-hang-guard` for detection
3. **Resource conflicts**: Use `make test-forked` for isolation

### Configuration Issues
1. **Plugin conflicts**: Check `pyproject.toml` configuration
2. **Marker issues**: Verify test markers are properly defined
3. **Environment problems**: Check environment variable configuration

## Contributing to Testing

### Adding New Tests
1. **Categorize appropriately**: Use appropriate markers
2. **Consider performance**: Keep tests fast and efficient
3. **Update documentation**: Document new test categories

### Improving Test Infrastructure
1. **Performance optimization**: Optimize slow tests
2. **Parallel compatibility**: Ensure tests work in parallel
3. **Resource management**: Optimize resource usage

### Reporting Issues
1. **Performance regressions**: Use `make test-perf` to document
2. **Reliability issues**: Document flaky or hanging tests
3. **Configuration problems**: Report configuration issues

## Conclusion

The testing setup delivers faster feedback, robust hang protection, and rich tooling for debugging and CI scaling. Use `make help` to discover commands, and lean on `make test-fast` during development. For questions or issues, see the troubleshooting guide above or open an issue.
