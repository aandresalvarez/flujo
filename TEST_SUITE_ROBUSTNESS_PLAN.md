# Test Suite Robustness Plan

## 🎯 Overview
This document outlines a comprehensive plan to make the Flujo test suite more robust, maintainable, and future-proof based on the issues we encountered.

## 🔧 Immediate Fixes (Priority 1)

### 1. Fix Collection Warnings
**Problem**: Pytest can't collect classes with `__init__` constructors
**Solution**: Add explicit `__init__` methods to test context classes

```python
# Before
class TestContext(BaseModel):
    counter: int = 0

# After
class TestContext(BaseModel):
    counter: int = 0

    def __init__(self, **data):
        super().__init__(**data)
```

**Files to fix**:
- `tests/application/core/test_executor_core_loop_step_migration.py`
- `tests/integration/test_executor_core_loop_conditional_migration.py`
- `tests/unit/test_ultra_executor_v2.py`

### 2. Add Test Timeouts
**Problem**: Tests can hang indefinitely
**Solution**: Add timeout decorators to all tests

```python
import pytest
import asyncio

@pytest.mark.asyncio
@pytest.mark.timeout(30)  # 30 second timeout
async def test_something():
    # test code
```

### 3. Improve Resource Management
**Problem**: Resource leaks in parallel execution
**Solution**: Add proper cleanup and resource limits

```python
@pytest.fixture(autouse=True)
def cleanup_resources():
    yield
    # Cleanup code
    gc.collect()
```

## 🏗️ Infrastructure Improvements (Priority 2)

### 1. Enhanced Makefile Targets

```makefile
# Add new robust test targets
.PHONY: test-robust
test-robust: .uv ## Run tests with enhanced robustness
	@echo "🛡️ Running robust test suite..."
	@CI=1 uv run pytest tests/ -m "not slow and not serial and not benchmark" \
		-n 4 --tb=short -q --timeout=60 --maxfail=5

.PHONY: test-stress
test-stress: .uv ## Run stress tests to identify resource issues
	@echo "💪 Running stress tests..."
	@CI=1 uv run pytest tests/ -m "stress" --timeout=300

.PHONY: test-memory
test-memory: .uv ## Run memory leak detection tests
	@echo "🧠 Running memory leak tests..."
	@CI=1 uv run pytest tests/ -m "memory" --timeout=120
```

### 2. Test Configuration Management

Create `pytest.ini` or enhance `pyproject.toml`:

```toml
[tool.pytest.ini_options]
timeout = 60
maxfail = 5
xfail_strict = true
addopts = [
    "--strict-markers",
    "--disable-warnings",
    "--tb=short",
    "--maxfail=5",
    "--timeout=60"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "stress: marks tests as stress tests",
    "memory: marks tests as memory leak tests",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "benchmark: marks tests as benchmark tests"
]
```

### 3. Resource Monitoring

```python
# tests/conftest.py
import psutil
import pytest
import gc

@pytest.fixture(autouse=True)
def monitor_resources():
    """Monitor resource usage during tests."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    initial_cpu = process.cpu_percent()

    yield

    # Check for resource leaks
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    if memory_increase > 100 * 1024 * 1024:  # 100MB
        pytest.fail(f"Memory leak detected: {memory_increase / 1024 / 1024:.2f}MB increase")
```

## 🧪 Test Quality Improvements (Priority 3)

### 1. Add Test Categories and Markers

```python
# tests/unit/test_example.py
import pytest

@pytest.mark.unit
@pytest.mark.fast
async def test_fast_unit_test():
    """Fast unit test."""
    pass

@pytest.mark.integration
@pytest.mark.slow
async def test_slow_integration_test():
    """Slow integration test."""
    pass

@pytest.mark.stress
@pytest.mark.memory
async def test_memory_stress_test():
    """Memory stress test."""
    pass
```

### 2. Implement Test Isolation

```python
# tests/conftest.py
import pytest
import asyncio
import tempfile
import shutil

@pytest.fixture(autouse=True)
def isolate_test_environment():
    """Ensure each test runs in isolation."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Reset asyncio event loop
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass
```

### 3. Add Performance Baselines

```python
# tests/performance/test_baselines.py
import pytest
import time
import statistics

class PerformanceBaseline:
    """Track performance baselines for tests."""

    def __init__(self):
        self.baselines = {}

    def record(self, test_name: str, duration: float):
        """Record test duration."""
        if test_name not in self.baselines:
            self.baselines[test_name] = []
        self.baselines[test_name].append(duration)

    def get_baseline(self, test_name: str) -> float:
        """Get baseline duration for a test."""
        if test_name not in self.baselines:
            return 1.0  # Default 1 second
        return statistics.median(self.baselines[test_name])

@pytest.fixture(scope="session")
def performance_baseline():
    """Performance baseline fixture."""
    return PerformanceBaseline()

@pytest.fixture(autouse=True)
def measure_performance(request, performance_baseline):
    """Measure test performance."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time

    test_name = request.node.name
    baseline = performance_baseline.get_baseline(test_name)

    # Fail if test is 10x slower than baseline
    if duration > baseline * 10:
        pytest.fail(f"Test {test_name} is {duration/baseline:.1f}x slower than baseline")
```

## 🔍 Monitoring and Alerting (Priority 4)

### 1. Test Health Dashboard

```python
# scripts/test_health_monitor.py
import json
import datetime
from pathlib import Path

class TestHealthMonitor:
    """Monitor test suite health."""

    def __init__(self):
        self.health_file = Path("test_health.json")
        self.load_health_data()

    def load_health_data(self):
        """Load historical health data."""
        if self.health_file.exists():
            with open(self.health_file) as f:
                self.health_data = json.load(f)
        else:
            self.health_data = {"runs": []}

    def record_run(self, passed: int, failed: int, skipped: int, duration: float):
        """Record test run results."""
        run_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration": duration,
            "total": passed + failed + skipped,
            "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 0
        }

        self.health_data["runs"].append(run_data)

        # Keep only last 100 runs
        if len(self.health_data["runs"]) > 100:
            self.health_data["runs"] = self.health_data["runs"][-100:]

        self.save_health_data()

    def save_health_data(self):
        """Save health data to file."""
        with open(self.health_file, "w") as f:
            json.dump(self.health_data, f, indent=2)

    def get_health_report(self) -> dict:
        """Generate health report."""
        if not self.health_data["runs"]:
            return {"status": "no_data"}

        recent_runs = self.health_data["runs"][-10:]
        avg_success_rate = sum(r["success_rate"] for r in recent_runs) / len(recent_runs)
        avg_duration = sum(r["duration"] for r in recent_runs) / len(recent_runs)

        return {
            "status": "healthy" if avg_success_rate > 0.95 else "degraded",
            "avg_success_rate": avg_success_rate,
            "avg_duration": avg_duration,
            "recent_runs": len(recent_runs)
        }
```

### 2. Automated Test Analysis

```python
# scripts/analyze_test_suite.py
import subprocess
import json
from pathlib import Path

def analyze_test_suite():
    """Analyze test suite for potential issues."""

    # Collect test information
    result = subprocess.run([
        "uv", "run", "pytest", "tests/", "--collect-only", "-q"
    ], capture_output=True, text=True)

    # Parse test collection
    tests = []
    for line in result.stdout.split('\n'):
        if '::' in line and 'test_' in line:
            tests.append(line.strip())

    # Analyze test distribution
    analysis = {
        "total_tests": len(tests),
        "unit_tests": len([t for t in tests if "/unit/" in t]),
        "integration_tests": len([t for t in tests if "/integration/" in t]),
        "benchmark_tests": len([t for t in tests if "/benchmarks/" in t]),
        "slow_tests": 0,  # Would need to check markers
        "fast_tests": 0
    }

    # Generate recommendations
    recommendations = []

    if analysis["unit_tests"] < analysis["total_tests"] * 0.6:
        recommendations.append("Consider adding more unit tests")

    if analysis["integration_tests"] > analysis["total_tests"] * 0.4:
        recommendations.append("Consider reducing integration test count")

    return {
        "analysis": analysis,
        "recommendations": recommendations
    }
```

## 🚀 Implementation Roadmap

### Phase 1: Immediate Fixes (Week 1)
- [ ] Fix all collection warnings
- [ ] Add timeout decorators to critical tests
- [ ] Implement basic resource monitoring
- [ ] Update Makefile with robust targets

### Phase 2: Infrastructure (Week 2)
- [ ] Implement test isolation
- [ ] Add performance baselines
- [ ] Create test health monitoring
- [ ] Set up automated test analysis

### Phase 3: Quality Assurance (Week 3)
- [ ] Add comprehensive test markers
- [ ] Implement stress testing framework
- [ ] Create memory leak detection
- [ ] Add test coverage reporting

### Phase 4: Monitoring (Week 4)
- [ ] Set up test health dashboard
- [ ] Implement automated alerts
- [ ] Create test performance tracking
- [ ] Add regression detection

## 📊 Success Metrics

### Test Reliability
- **Target**: 99.5% test pass rate
- **Current**: 100% (2251/2251 passed)
- **Measurement**: Track over 30 days

### Test Performance
- **Target**: <60 seconds for fast test suite
- **Current**: 56.31 seconds
- **Measurement**: Monitor execution time trends

### Resource Usage
- **Target**: <100MB memory increase per test run
- **Current**: Within limits
- **Measurement**: Monitor memory usage patterns

### Test Coverage
- **Target**: >90% code coverage
- **Current**: Unknown
- **Measurement**: Implement coverage reporting

## 🔄 Continuous Improvement

### Weekly Reviews
- Review test health dashboard
- Analyze performance trends
- Identify flaky tests
- Update baselines

### Monthly Assessments
- Evaluate test suite effectiveness
- Identify areas for improvement
- Update test strategies
- Plan new test categories

### Quarterly Planning
- Assess test suite scalability
- Plan for new features
- Update test infrastructure
- Review monitoring tools

## 🛡️ Best Practices

### Test Design
1. **Isolation**: Each test should be independent
2. **Determinism**: Tests should be repeatable
3. **Speed**: Fast tests should complete in <1 second
4. **Clarity**: Test names should be descriptive
5. **Maintenance**: Tests should be easy to update

### Test Organization
1. **Categories**: Use markers to organize tests
2. **Hierarchy**: Unit → Integration → E2E
3. **Parallelization**: Design tests for parallel execution
4. **Resource Management**: Clean up after tests
5. **Error Handling**: Graceful failure handling

### Monitoring
1. **Health Checks**: Regular test suite health monitoring
2. **Performance Tracking**: Monitor execution times
3. **Resource Monitoring**: Track memory and CPU usage
4. **Alerting**: Notify on test failures
5. **Reporting**: Generate comprehensive reports

This plan will make the test suite robust, maintainable, and future-proof! 🎉
