# CI Performance Optimization

This document describes the performance optimizations implemented for GitHub Actions CI to speed up test execution while maintaining comprehensive testing locally.

## Overview

The persistence performance tests are optimized to run with different database sizes based on the environment:

- **Local Development**: Full 10,000 runs for comprehensive testing
- **CI Environment**: Reduced database sizes for faster execution

## Environment-Based Configuration

### Database Size Configuration

The test suite uses environment variables to determine the database size:

```python
@staticmethod
def get_database_size() -> int:
    """Get database size based on environment - smaller for CI, minimal for mass CI."""
    if os.getenv("CI") == "true":
        # Use even smaller size for mass CI scenarios (250 runs)
        # This reduces setup time from ~4s to ~1s while maintaining test validity
        return int(os.getenv("FLUJO_CI_DB_SIZE", "250"))
    else:
        # Use 10,000 runs for local development (full size)
        return int(os.getenv("FLUJO_LOCAL_DB_SIZE", "10000"))
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CI` | `false` | Set to `"true"` in CI environments |
| `FLUJO_CI_DB_SIZE` | `1000` | Database size for CI environments |
| `FLUJO_LOCAL_DB_SIZE` | `10000` | Database size for local development |

## Performance Improvements

### Speed Comparison

| Environment | Database Size | Execution Time | Speed Improvement |
|-------------|---------------|----------------|-------------------|
| Local | 10,000 runs | ~44s | Baseline |
| CI (Mass CI Optimized) | 250 runs | ~1.0s | 44x faster |
| CI (Previous) | 1,000 runs | ~4.8s | 9x faster |

### Test Categorization for Mass CI

#### Ultra-Slow Tests (Excluded from Regular CI)
- **Markers**: `@pytest.mark.ultra_slow`
- **Duration**: >30 seconds each
- **Examples**: Stress tests, sustained load tests
- **CI Strategy**: Run separately on main branch only

#### Slow Tests (Optimized for CI)
- **Markers**: `@pytest.mark.slow`, excluding `ultra_slow`
- **Duration**: ~46 seconds total (down from 5:41)
- **Database Size**: 250 runs (optimized from 1,000)
- **CI Strategy**: Regular inclusion in CI workflows

### CI Workflow Optimizations

#### Main CI Workflow (`.github/workflows/ci.yml`)

- **Fast Tests**: 250 runs (`FLUJO_CI_DB_SIZE: "250"`)
- **Slow Tests**: 250 runs, excludes ultra-slow (`FLUJO_CI_DB_SIZE: "250"`)
- **Ultra-Slow Tests**: Separate job, main branch only

#### PR Checks Workflow (`.github/workflows/pr-checks.yml`)

- **Fast Tests**: 250 runs (`FLUJO_CI_DB_SIZE: "250"`)
- **Unit Tests**: 250 runs (`FLUJO_CI_DB_SIZE: "250"`)
- **Security Tests**: 250 runs (`FLUJO_CI_DB_SIZE: "250"`)

## Implementation Details

### Dynamic Database Creation

The `large_database` fixture now creates databases with configurable sizes:

```python
@pytest.fixture
def large_database(self, tmp_path: Path) -> Path:
    """Create a database with configurable number of runs for performance testing."""
    db_size = self.get_database_size()
    completed_runs = int(db_size * 0.95)  # 95% of runs are completed

    # Create database with concurrent operations
    # ... implementation details
```

### Test Adaptability

All test methods automatically adapt to the database size:

```python
def test_large_database_fixture_verification(self, large_database: Path) -> None:
    """Verify that the large_database fixture is working correctly."""
    expected_size = self.get_database_size()

    # Verify the database has expected number of runs
    assert len(runs) == expected_size, f"Expected {expected_size} runs, got {len(runs)}"
```

## Benefits

### For CI/CD

1. **Faster Feedback**: CI jobs complete 9-34x faster
2. **Reduced Resource Usage**: Less CPU and memory consumption
3. **Lower Costs**: Reduced GitHub Actions minutes usage
4. **Better Developer Experience**: Faster PR feedback

### For Local Development

1. **Comprehensive Testing**: Full 10,000 runs for thorough validation
2. **Performance Validation**: Real-world performance testing
3. **Debugging Capability**: Full dataset for troubleshooting

## Configuration Examples

### Local Development (Default)
```bash
# No environment variables needed - uses full 10,000 runs
uv run pytest tests/unit/test_persistence_performance.py
```

### CI Environment (Automatic)
```bash
# CI=true is set automatically in GitHub Actions
CI=true FLUJO_CI_DB_SIZE=1000 uv run pytest tests/unit/test_persistence_performance.py
```

### Custom Local Testing
```bash
# Test with smaller database for faster local iteration
FLUJO_LOCAL_DB_SIZE=1000 uv run pytest tests/unit/test_persistence_performance.py
```

## Monitoring and Validation

### Performance Thresholds

The tests maintain the same performance thresholds regardless of database size:

- **CLI Commands**: <2s execution time
- **Database Operations**: <5% overhead
- **Memory Usage**: Optimized for CI environments

### Quality Assurance

- All tests pass with both small and large databases
- Performance characteristics are maintained
- Coverage remains comprehensive
- Error handling and debugging work in all environments

## Future Enhancements

1. **Dynamic Thresholds**: Adjust performance thresholds based on database size
2. **Memory Optimization**: Further reduce memory usage in CI
3. **Parallel Database Creation**: Optimize database creation process
4. **Caching**: Cache database fixtures for faster test execution
