# Timeout and Linger Issues Analysis

## Executive Summary

**Problem**: 4 test files are experiencing timeout (>180s) or linger (passes but takes >180s) issues:
1. ⏰ `test_cli_performance_edge_cases.py` - **TIMEOUT** (>180s)
2. ✅⏳ `test_sqlite_concurrency_edge_cases.py` - **PASS_LINGER** (181.03s)
3. ✅⏳ `test_sqlite_fault_tolerance.py` - **PASS_LINGER** (181.02s)
4. ✅⏳ `test_sqlite_retry_mechanism.py` - **PASS_LINGER** (181.06s)

**Impact**: These tests are significantly slowing down the CI pipeline and one is causing failures.

---

## Detailed Analysis

### 1. ⏰ `test_cli_performance_edge_cases.py` - TIMEOUT (Critical)

**Status**: Test times out at 180s (per-test timeout is 60s, outer timeout is 180s)

**Markers**: `@pytest.mark.veryslow`, `@pytest.mark.serial`

**Root Causes**:

#### A. Large Database Creation in Fixture
```python
@pytest.fixture(scope="module")
async def large_database_with_mixed_data(self, tmp_path_factory) -> Path:
    total = int(_os.getenv("FLUJO_CI_DB_SIZE", "200"))  # Default 200 runs
    for i in range(total):
        # Create run start
        await backend.save_run_start(...)
        # Create run end
        await backend.save_run_end(...)
        # Add step data (2 steps per completed run)
        if status == "completed":
            for step_idx in range(2):
                await backend.save_step_result(...)
```

**Problem**: 
- Creates 200 runs (or `FLUJO_CI_DB_SIZE` if set)
- Each run involves 1-3 async database writes
- Total: ~400-600 async operations in fixture alone
- In CI, `FLUJO_CI_DB_SIZE` was recently set to 50 (line 13 of `.github/workflows/pr-checks.yml`)

#### B. Multiple CLI Invocations
The test file contains ~10 test methods, each invoking the CLI multiple times:
```python
def test_lens_list_with_various_filters(...):
    filter_tests = [
        ["lens", "list", "--status", "completed"],
        ["lens", "list", "--status", "failed"],
        # ... 4 more filter combinations
    ]
    for filter_args in filter_tests:
        result = runner.invoke(app, filter_args)
```

**Total Operations**: 
- Fixture: ~200-600 DB operations
- Tests: ~50-100 CLI invocations
- Each CLI invocation: Opens DB, queries, closes connection
- **Estimated Time**: 120-200+ seconds

**Why It Times Out**:
1. **Module-scoped async fixture** with `tmp_path_factory` - may not execute properly
2. **Serial execution** prevents parallelization
3. **No connection pooling** - each CLI invocation opens/closes DB
4. **Large dataset** - 200 runs with nested data

---

### 2. ✅⏳ `test_sqlite_concurrency_edge_cases.py` - PASS_LINGER (181.03s)

**Status**: Passes but takes 181.03s (just over the 180s threshold)

**Markers**: `@pytest.mark.serial`

**Root Causes**:

#### A. Heavy Concurrent Operations
```python
@pytest.mark.asyncio
async def test_concurrent_performance_under_load(...):
    """Test performance under concurrent load."""
    # Creates many concurrent tasks
```

**Problem**:
- 9 test methods, all async
- Tests involve deliberate concurrent access patterns
- Uses `asyncio.gather()` for parallel operations
- Intentionally stresses the SQLite backend

#### B. Serial Execution Requirement
- Marked `@pytest.mark.serial` to avoid DB conflicts
- **Cannot parallelize** these tests
- All 9 tests run sequentially

**Estimated Breakdown**:
- 9 tests × ~20s each = 180s
- Overhead (setup/teardown) = 1-2s
- **Total**: 181s

**Why It Lingers**:
- Tests are intentionally slow (testing concurrency limits)
- Serial execution compounds the duration
- Just barely exceeds the 180s linger threshold

---

### 3. ✅⏳ `test_sqlite_fault_tolerance.py` - PASS_LINGER (181.02s)

**Status**: Passes but takes 181.02s

**Markers**: `@pytest.mark.serial`

**Root Causes**:

#### A. Fault Injection and Recovery Testing
```python
@pytest.mark.asyncio
async def test_sqlite_backend_handles_corrupted_database(...):
    """Test that SQLiteBackend can handle corrupted database files."""
    # Deliberately corrupts DB, then tests recovery
```

**Problem**:
- 15 test methods, all async
- Each test deliberately introduces faults (corruption, disk errors, etc.)
- Tests recovery mechanisms with retries
- Each recovery involves multiple retry attempts with backoff

#### B. Retry Mechanisms with Delays
```python
# Internal retry logic in SQLiteBackend
max_retries = 3
for attempt in range(max_retries):
    try:
        # Operation
    except OperationalError:
        if attempt < max_retries - 1:
            await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
```

**Estimated Breakdown**:
- 15 tests × ~12s each (with retry delays) = 180s
- Overhead = 1-2s
- **Total**: 181s

**Why It Lingers**:
- Intentional delays for retry testing
- Corruption/recovery scenarios are inherently slow
- Serial execution prevents parallelization

---

### 4. ✅⏳ `test_sqlite_retry_mechanism.py` - PASS_LINGER (181.06s)

**Status**: Passes but takes 181.06s

**Markers**: `@pytest.mark.serial`

**Root Causes**:

#### A. Focused Retry Testing
```python
@pytest.mark.asyncio
async def test_with_retries_max_retries_exceeded(...):
    """Test that max retries are respected for database locked errors."""
    # Forces max retries to be hit
```

**Problem**:
- 16 test methods, all async
- Each test specifically tests retry logic
- Involves intentional failures and retries
- Multiple retry attempts with delays

#### B. Concurrent Safety Testing
```python
@pytest.mark.asyncio
async def test_with_retries_concurrent_access_safety(...):
    """Test that retry mechanism is safe under concurrent access."""
    # Creates concurrent access patterns
```

**Estimated Breakdown**:
- 16 tests × ~11s each = 176s
- Overhead (especially for concurrent tests) = 5s
- **Total**: 181s

**Why It Lingers**:
- Retry delays accumulate
- Concurrent safety tests are slow
- Serial execution required for accurate retry testing

---

## Solutions and Recommendations

### Immediate Fixes (High Priority)

#### 1. Fix `test_cli_performance_edge_cases.py` TIMEOUT

**Option A: Convert Module Fixture to Function-Scoped** (Recommended)
```python
@pytest.fixture
async def large_database_with_mixed_data(self, tmp_path) -> Path:
    """Create a database - function scoped for proper async execution."""
    db_path = tmp_path / "mixed_ops.db"
    backend = SQLiteBackend(db_path)
    
    # Reduce default size for unit tests
    total = int(os.getenv("FLUJO_CI_DB_SIZE", "50"))  # Changed from 200
    
    # ... rest of fixture
```

**Benefits**:
- Proper async execution
- Smaller default dataset (50 vs 200)
- More predictable timing

**Option B: Add pytest-timeout Override**
```python
pytestmark = [
    pytest.mark.veryslow,
    pytest.mark.serial,
    pytest.mark.timeout(300)  # 5 minutes for entire module
]
```

**Option C: Split Into Multiple Test Files**
- `test_cli_performance_list.py` - List operations only
- `test_cli_performance_show.py` - Show operations only
- `test_cli_performance_filters.py` - Filter operations only

Each file would be faster and more focused.

**Recommendation**: Implement **Option A + Option B** together.

---

#### 2. Address SQLite Test Linger Issues

**Option A: Mark as `@pytest.mark.slow`** (Recommended for CI)
```python
# In each lingering test file
pytestmark = [
    pytest.mark.serial,
    pytest.mark.slow  # Add this to exclude from fast/standard test runs
]
```

**Option B: Increase Linger Threshold**
```python
# In test runner configuration or CI
LINGER_THRESHOLD = 240  # Increase from 180s to 240s (4 minutes)
```

**Option C: Optimize Test Execution**

For `test_sqlite_concurrency_edge_cases.py`:
```python
# Reduce concurrent load in tests
async def test_concurrent_performance_under_load(...):
    # Old: 100 concurrent operations
    num_operations = 50  # Reduce to 50
    
    # Old: Test with 20 concurrent clients
    num_clients = 10  # Reduce to 10
```

For `test_sqlite_fault_tolerance.py`:
```python
# Reduce retry attempts in tests
# Old: max_retries = 5
max_retries = 3  # Reduce retry attempts

# Reduce sleep delays
# Old: await asyncio.sleep(0.5)
await asyncio.sleep(0.1)  # Shorter delays
```

For `test_sqlite_retry_mechanism.py`:
```python
# Similar optimizations: reduce retries and delays
```

**Recommendation**: Implement **Option A** (mark as slow) for immediate CI relief, then implement **Option C** (optimizations) for long-term improvement.

---

### Medium-Term Improvements

#### 3. Test Suite Organization

**Current State**:
- All 4 tests are in unit/integration folders
- All marked `serial` (no parallelization)
- Mixed performance/concurrency/fault-tolerance concerns

**Proposed Organization**:
```
tests/
  fast/              # <5s per test
  standard/          # <30s per test
  slow/              # <180s per test
    test_sqlite_concurrency.py
    test_sqlite_fault_tolerance.py
    test_sqlite_retry_mechanism.py
  veryslow/          # >180s per test
    test_cli_performance_edge_cases.py
```

**CI Configuration**:
```yaml
# Fast tests - always run
- name: Fast tests
  run: pytest tests/fast/ -n 4

# Standard tests - always run  
- name: Standard tests
  run: pytest tests/standard/ -n 4

# Slow tests - run on PR only
- name: Slow tests
  run: pytest tests/slow/ -n 1
  if: github.event_name == 'pull_request'

# Very slow tests - run nightly only
- name: Very slow tests
  run: pytest tests/veryslow/
  if: github.event_name == 'schedule'
```

---

#### 4. Database Connection Pooling

**Problem**: CLI tests open/close DB for each invocation.

**Solution**: Implement connection pooling in SQLiteBackend:
```python
class SQLiteBackend:
    _connection_pool: ClassVar[dict[str, Connection]] = {}
    
    async def _get_connection(self) -> Connection:
        """Get pooled connection or create new one."""
        key = str(self.db_path)
        if key not in self._connection_pool:
            self._connection_pool[key] = await aiosqlite.connect(...)
        return self._connection_pool[key]
```

**Benefits**:
- Reduces connection overhead
- Speeds up CLI tests significantly
- Maintains compatibility

---

#### 5. Parameterize Test Scales

**Current**: Hard-coded test scales
**Proposed**: Environment-based scaling

```python
# In conftest.py or each test file
TEST_SCALE = os.getenv("FLUJO_TEST_SCALE", "small")

SCALE_CONFIG = {
    "small": {"db_size": 20, "concurrent_ops": 10, "retries": 2},
    "medium": {"db_size": 100, "concurrent_ops": 50, "retries": 3},
    "large": {"db_size": 500, "concurrent_ops": 200, "retries": 5},
}

config = SCALE_CONFIG[TEST_SCALE]
```

**Usage**:
```bash
# CI fast tests
FLUJO_TEST_SCALE=small pytest tests/

# Nightly comprehensive tests
FLUJO_TEST_SCALE=large pytest tests/
```

---

### Long-Term Strategic Changes

#### 6. Separate Performance Benchmarks

**Current**: Performance tests mixed with unit tests.

**Proposed**: Move to dedicated benchmark suite:
```
tests/
  benchmarks/
    test_cli_performance.py
    test_sqlite_performance.py
```

**Use pytest-benchmark**:
```python
def test_lens_list_performance(benchmark, large_db):
    """Benchmark lens list command."""
    result = benchmark(runner.invoke, app, ["lens", "list"])
    assert result.exit_code == 0
```

**Benefits**:
- Clear separation of concerns
- Dedicated CI job for benchmarks
- Historical performance tracking

---

#### 7. Mock Heavy Operations in Unit Tests

**Current**: Unit tests create real databases and perform real operations.

**Proposed**: Mock heavy operations, use real DB only in integration tests:

```python
# Unit test - fast
@pytest.mark.unit
def test_lens_list_formatting(mocker):
    """Test lens list output formatting (mocked data)."""
    mocker.patch('flujo.state.backends.sqlite.SQLiteBackend.get_runs',
                 return_value=[mock_run_1, mock_run_2])
    result = runner.invoke(app, ["lens", "list"])
    assert "run_001" in result.output

# Integration test - slow (runs in slow suite)
@pytest.mark.slow
@pytest.mark.integration
def test_lens_list_with_real_db(real_db):
    """Test lens list with real database."""
    result = runner.invoke(app, ["lens", "list"])
    assert result.exit_code == 0
```

---

## Implementation Plan

### Phase 1: Immediate Fixes (This PR)

**Priority: CRITICAL - Fix TIMEOUT**

1. **Fix `test_cli_performance_edge_cases.py`**:
   - [ ] Convert module fixture to function-scoped
   - [ ] Reduce default DB size to 50
   - [ ] Add `@pytest.mark.timeout(300)`
   - [ ] Test locally to verify <180s execution

2. **Mark Linger Tests as Slow**:
   - [ ] Add `@pytest.mark.slow` to all 3 SQLite test files
   - [ ] Update CI to exclude slow tests from standard runs
   - [ ] Keep slow tests for PR/nightly builds

**Estimated Impact**: 
- TIMEOUT eliminated ✅
- CI fast tests: ~30-60s faster
- Slow tests still run in appropriate contexts

---

### Phase 2: Optimization (Next PR)

**Priority: HIGH - Improve test performance**

1. **Optimize SQLite Tests**:
   - [ ] Reduce concurrent operations (100 → 50)
   - [ ] Reduce retry counts (5 → 3)
   - [ ] Reduce sleep delays (0.5s → 0.1s)
   - [ ] Target: <150s per test file

2. **Implement Test Scale Configuration**:
   - [ ] Add `FLUJO_TEST_SCALE` environment variable
   - [ ] Create scale configs (small/medium/large)
   - [ ] Update tests to use scale config
   - [ ] Update CI to use "small" scale

**Estimated Impact**:
- Linger tests: 181s → 120-140s
- More flexible test execution
- Better CI/local development balance

---

### Phase 3: Refactoring (Future PR)

**Priority: MEDIUM - Long-term improvements**

1. **Reorganize Test Suite**:
   - [ ] Create fast/standard/slow/veryslow directories
   - [ ] Move tests to appropriate directories
   - [ ] Update CI to run appropriate suites
   - [ ] Document test organization

2. **Extract Performance Benchmarks**:
   - [ ] Create `tests/benchmarks/` directory
   - [ ] Move performance tests to benchmarks
   - [ ] Set up pytest-benchmark
   - [ ] Create dedicated benchmark CI job

3. **Implement Connection Pooling**:
   - [ ] Add connection pooling to SQLiteBackend
   - [ ] Update tests to use pooled connections
   - [ ] Measure performance improvement

**Estimated Impact**:
- Much cleaner test organization
- Faster local development cycles
- Better CI pipeline efficiency

---

## Summary Table

| Test File | Current Status | Time | Issue | Phase 1 Fix | Expected Time After Fix |
|-----------|----------------|------|-------|-------------|------------------------|
| `test_cli_performance_edge_cases.py` | ⏰ TIMEOUT | >180s | Critical | Fix fixture + reduce size + timeout | <120s |
| `test_sqlite_concurrency_edge_cases.py` | ✅⏳ LINGER | 181.03s | High | Mark as slow, exclude from fast CI | 181s (in slow suite) |
| `test_sqlite_fault_tolerance.py` | ✅⏳ LINGER | 181.02s | High | Mark as slow, exclude from fast CI | 181s (in slow suite) |
| `test_sqlite_retry_mechanism.py` | ✅⏳ LINGER | 181.06s | High | Mark as slow, exclude from fast CI | 181s (in slow suite) |

**After Phase 1**: 
- ✅ No timeouts
- ✅ Fast CI (<2 min for fast suite)
- ✅ Slow tests run in appropriate contexts

**After Phase 2**:
- ✅ All tests <150s
- ✅ Configurable test scales
- ✅ Better performance

**After Phase 3**:
- ✅ Clean test organization
- ✅ Dedicated benchmarks
- ✅ Optimized infrastructure

---

## Recommendation

**Start with Phase 1 immediately** to fix the critical TIMEOUT and move linger tests out of the fast CI path. This will unblock the CI pipeline and allow development to continue smoothly.

Phase 2 and 3 can be implemented incrementally as time permits, with clear measurable improvements at each stage.

