# CI Performance Guide

This guide explains the CI/CD performance optimizations implemented for the Flujo project and how the workflows are structured for maximum efficiency.

## Overview

The CI/CD pipeline has been optimized for faster feedback through:

1. **Separate Workflows** - Different workflows for different scenarios
2. **Parallel Test Execution** - Using pytest-xdist in CI
3. **Smart Test Categorization** - Running fast tests first
4. **Optimized Matrix Strategies** - Reducing redundant test runs
5. **Caching Strategies** - Leveraging GitHub Actions caching

## Workflow Structure

### 1. PR Checks (Fast Feedback)
**File**: `.github/workflows/pr-checks.yml`
**Trigger**: Pull requests to main
**Goal**: Provide fast feedback for developers

#### Jobs:
- **Quality Checks**: Lint and type checking
- **Fast Tests**: Parallel execution of fast tests (~25s)
- **Unit Tests**: Ultra-fast unit test feedback
- **Security Tests**: Critical security validation
- **Coverage Report**: Combined coverage from fast tests

#### Performance Characteristics:
- **Total Time**: ~2-3 minutes
- **Parallel Jobs**: 4 concurrent jobs
- **Fail-Fast**: Stops on first failure for quick feedback

### 2. Main CI (Comprehensive)
**File**: `.github/workflows/ci.yml`
**Trigger**: Pushes to main branch
**Goal**: Comprehensive testing and coverage

#### Jobs:
- **Quality Checks**: Lint and type checking
- **Fast Tests**: Parallel execution on Python 3.11, 3.12
- **Slow Tests**: Serial execution on Python 3.11 only
- **Full Test Suite**: Legacy tests on Python 3.9, 3.10
- **Coverage Report**: Combined coverage from all tests
- **Performance Analysis**: Optional performance monitoring

#### Performance Characteristics:
- **Total Time**: ~8-12 minutes
- **Comprehensive Coverage**: All Python versions and test types
- **Performance Monitoring**: Tracks test performance over time

## Performance Improvements

### Before Optimization
- **Single workflow** for all scenarios
- **All tests run serially** on all Python versions
- **No test categorization**
- **PR feedback time**: ~15-20 minutes
- **Main branch feedback time**: ~20-25 minutes

### After Optimization
- **Separate workflows** for different scenarios
- **Parallel test execution** with pytest-xdist
- **Smart test categorization** (fast/slow/serial)
- **PR feedback time**: ~2-3 minutes (85% improvement)
- **Main branch feedback time**: ~8-12 minutes (50% improvement)

## Test Execution Strategy

### Fast Tests (Parallel)
- **Execution**: 8 workers in parallel
- **Duration**: ~25 seconds
- **Coverage**: Unit tests, integration tests (excluding slow ones)
- **Use Case**: Primary feedback for PRs

### Slow Tests (Serial)
- **Execution**: Single worker (serial)
- **Duration**: ~3-4 minutes
- **Coverage**: Performance tests, benchmarks, serial tests
- **Use Case**: Comprehensive validation on main branch

### Unit Tests (Ultra Fast)
- **Execution**: 8 workers in parallel
- **Duration**: ~10-15 seconds
- **Coverage**: Unit tests only
- **Use Case**: Ultra-fast feedback for unit test changes

## Matrix Optimization

### PR Workflow
- **Python versions**: 3.11 (fast tests), 3.12 (unit tests)
- **Strategy**: Minimal matrix for speed
- **Fail-fast**: Enabled for quick feedback

### Main CI Workflow
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Strategy**: Comprehensive matrix for thorough testing
- **Fail-fast**: Disabled for complete coverage

## Caching Strategy

### Dependency Caching
```yaml
- name: Set up uv
  uses: astral-sh/setup-uv@v1
  with:
    enable-cache: true
```

### Coverage Artifact Caching
- Fast tests coverage stored separately
- Slow tests coverage stored separately
- Combined coverage generated from all sources

## Performance Monitoring

### Automated Performance Analysis
- **Job**: `test-performance` in main CI
- **Trigger**: Only on pushes to main
- **Output**: Performance report artifact
- **Purpose**: Track test performance over time

### Performance Metrics
- Test execution times
- Slow test identification
- Coverage trends
- Resource usage patterns

## Best Practices

### For Developers
1. **Use PR workflow** for fast feedback during development
2. **Monitor CI times** to identify performance regressions
3. **Use appropriate test markers** for new tests
4. **Check coverage reports** for test quality

### For Maintainers
1. **Monitor main CI** for comprehensive validation
2. **Review performance reports** for trends
3. **Optimize slow tests** when identified
4. **Update matrix strategies** as needed

### For CI/CD
1. **Use appropriate workflow** for the scenario
2. **Leverage caching** for faster builds
3. **Monitor resource usage** for optimization
4. **Track performance metrics** over time

## Configuration

### Environment Variables
```yaml
env:
  FLUJO_CLI_PERF_THRESHOLD: "0.2"
  FLUJO_CV_THRESHOLD: "1.0"
```

### Python Version Strategy
- **PR workflow**: 3.11, 3.12 (speed focus)
- **Main workflow**: 3.9, 3.10, 3.11, 3.12 (comprehensive)

### Test Execution Commands
```bash
# Fast tests (parallel)
uv run coverage run --source=flujo --parallel-mode -m pytest tests/ -m "not slow and not serial and not benchmark" -n auto

# Slow tests (serial)
uv run coverage run --source=flujo --parallel-mode -m pytest tests/ -m "slow or serial or benchmark"

# Unit tests only (ultra-fast)
uv run pytest tests/unit/ -n auto
```

## Troubleshooting

### Common Issues

1. **Tests failing in parallel**
   - Check if tests are properly marked as `serial`
   - Ensure tests don't share resources
   - Review test isolation

2. **Slow CI execution**
   - Check cache hit rates
   - Review matrix strategy
   - Optimize slow tests

3. **Coverage issues**
   - Verify coverage collection from all jobs
   - Check artifact upload/download
   - Review coverage thresholds

4. **Resource constraints**
   - Monitor GitHub Actions minutes usage
   - Optimize test execution time
   - Consider test sharding

### Performance Tips

1. **Use appropriate workflow**:
   - PR workflow for development
   - Main workflow for releases

2. **Optimize test execution**:
   - Mark slow tests appropriately
   - Use parallel execution where possible
   - Leverage caching effectively

3. **Monitor performance**:
   - Track CI execution times
   - Review performance reports
   - Optimize based on trends

## Future Improvements

1. **Test Sharding** - Split tests across multiple workers
2. **Incremental Testing** - Only run affected tests
3. **Smart Caching** - Cache test results between runs
4. **Performance Regression Detection** - Automated alerts
5. **Resource Optimization** - Better resource utilization

## Metrics and Monitoring

### Key Metrics to Track
- **PR feedback time**: Target < 3 minutes
- **Main CI time**: Target < 12 minutes
- **Cache hit rate**: Target > 80%
- **Test execution time**: Track trends over time

### Monitoring Tools
- GitHub Actions timing reports
- Performance analysis artifacts
- Coverage trend analysis
- Resource usage monitoring

## Contributing

When modifying CI workflows:

1. **Test locally first** - Use `make test-fast` for quick validation
2. **Consider performance impact** - Measure before and after changes
3. **Update documentation** - Keep this guide current
4. **Monitor metrics** - Track the impact of changes

### Guidelines for New Tests
1. **Categorize appropriately** - Use correct pytest markers
2. **Consider execution time** - Keep tests fast when possible
3. **Test in CI** - Ensure tests work in CI environment
4. **Document changes** - Update relevant documentation
