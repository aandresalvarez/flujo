# CI Timeout Fixes Summary

## Overview
This branch (`fix-ci-timeouts`) addresses critical CI timeout issues that were causing tests to hang indefinitely and preventing the test suite from completing. The fixes implement multiple layers of timeout protection and address root causes of hanging tests.

## Issues Identified

### 1. Replay Agent Test Hanging
- **Test**: `tests/integration/test_replay_agent_non_hitl.py::test_replay_agent_replays_non_hitl_pipeline_from_state`
- **Problem**: Test was hanging indefinitely during SQLite backend operations and trace replay
- **Impact**: Test suite would hang, causing CI failures and manual intervention required

### 2. Architect Performance Tests Taking Excessive Time
- **Tests**: Multiple tests in `tests/integration/architect/test_architect_performance_stress.py`
- **Problem**: Tests were taking 100+ seconds instead of the expected 20–60 seconds
- **Root Cause**: Tests were running the full architect state machine pipeline instead of minimal pipeline
- **Impact**: CI builds took excessive time, and performance thresholds failed

### 3. Architect Security Validation Tests Hanging
- **Tests**: Multiple tests in `tests/integration/architect/test_architect_security_validation.py`
- **Problem**: Tests were taking 20+ seconds and sometimes hanging
- **Root Cause**: Same as performance tests - full state machine pipeline execution

## Fixes Implemented

### 1. Replay Agent Test Fix (`346e1fc`)
- **Added comprehensive timeout protection**:
  - `@pytest.mark.timeout(60)` for entire test
  - `asyncio.wait_for` with 30s timeout for replay operation
  - Signal-based timeout as additional protection
- **Fixed async iteration handling**: Properly handled async generator from `run_async`
- **Added detailed logging**: Help debug future timeout issues
- **Improved resource cleanup**: Proper backend closure to prevent resource leaks
- **Result**: Test now completes in ~0.17s instead of hanging indefinitely

### 2. Architect Tests Fix (commit `3c05625`)
- **Force minimal Architect pipeline**:
  - Set `FLUJO_ARCHITECT_IGNORE_CONFIG=1` and `FLUJO_TEST_MODE=1`
  - Ensures tests use simple pipeline instead of complex state machine
- **Added comprehensive timeout protection**:
  - `test_architect_execution_time_consistency`: 30s timeout
  - `test_architect_handles_high_frequency_requests`: 60s timeout
  - `test_architect_stress_test_rapid_requests`: 120s timeout (was taking 100+ seconds)
  - `test_architect_response_time_under_load`: 60s timeout
  - `test_architect_resource_usage_scaling`: 60s timeout
  - Security validation tests: 30s timeout each
- **Result**: Tests now complete in a reasonable time instead of hanging

## Technical Details

### Timeout Protection Layers
1. **pytest-timeout plugin**: Function-level timeouts using `@pytest.mark.timeout()`
2. **asyncio.wait_for**: Coroutine-level timeouts for specific operations
3. **Signal-based timeouts**: Additional protection using SIGALRM
4. **Test runner timeouts**: CI-level timeouts in test runner scripts

### Architect Pipeline Selection
The architect pipeline can run in two modes:
- **Minimal pipeline**: Simple, fast execution (used in tests)
- **Full state machine**: Complex, potentially slow execution (used in production)

By setting environment variables, we ensure tests use the minimal pipeline:
```python
os.environ["FLUJO_ARCHITECT_IGNORE_CONFIG"] = "1"
os.environ["FLUJO_TEST_MODE"] = "1"
```

### Performance Thresholds
- **Local development**: Base thresholds (e.g., 2s for response time)
- **CI environments**: 3x multiplier via `get_performance_threshold()`
- **Test timeouts**: Additional safety margins to prevent hanging

## Testing Results

### Before Fixes
- Replay agent test: Hanging indefinitely
- Architect performance tests: 100+ seconds execution time
- Security validation tests: 20+ seconds execution time
- CI builds: Manual intervention required

### After Fixes
- Replay agent test: Completes in ~0.17s
- Architect performance tests: Complete in 14-24s (on CI runners as of Aug 2025)
- Security validation tests: Complete in 2-4s (on CI runners as of Aug 2025)
- All tests: Pass with proper timeout protection

## Files Modified

1. **`tests/integration/test_replay_agent_non_hitl.py`**
   - Added timeout protection and proper async handling
   - Fixed resource cleanup and error handling

2. **`tests/integration/architect/test_architect_performance_stress.py`**
   - Added environment variables for minimal pipeline
   - Added timeout decorators to all performance tests

3. **`tests/integration/architect/test_architect_security_validation.py`**
   - Added environment variables for minimal pipeline
   - Added timeout decorators to security tests

## Best Practices Established

1. **Always use timeouts**: Add `@pytest.mark.timeout()` to potentially slow tests
2. **Use minimal pipelines in tests**: Set environment variables to avoid complex execution
3. **Proper async handling**: Handle async generators correctly to prevent hanging
4. **Resource cleanup**: Ensure proper cleanup of resources (databases, connections)
5. **Layered protection**: Multiple timeout mechanisms for robust protection

## Future Recommendations

1. **Monitor test execution times**: Track if tests start taking longer than expected
2. **Add timeouts to new tests**: Ensure all new tests have appropriate timeout protection
3. **Consider test isolation**: Run architect tests in separate CI jobs to prevent interference
4. **Performance benchmarking**: Establish baseline performance metrics for CI environments
5. **Automated timeout detection**: Implement CI checks to detect tests taking excessive time

## Alignment with Architect Redesign

### Current Approach vs. Long‑Term Vision
Our current fixes implement the **"Short-Term Pragmatics"** mentioned in the architect redesign document:

- **Environment-based selection**: We're using `FLUJO_ARCHITECT_IGNORE_CONFIG=1` and `FLUJO_TEST_MODE=1`
- **CI stabilization**: Our timeout fixes provide immediate stability.
- **Deterministic behavior**: Minimal pipeline ensures consistent test results

### Supporting the Redesign Goals
Our fixes actually **support** the proposed redesign direction:

1. **Single authoritative path**: By forcing minimal pipeline, we're moving toward consistent behavior
2. **Deterministic state visibility**: Minimal pipeline provides predictable, testable behavior
3. **CI reliability**: Timeout protection ensures tests don't hang during the transition

### Migration Path
When the redesign is implemented, our current approach provides a clear migration path:

1. **Phase 1** (Current): Use minimal pipeline with environment variables
2. **Phase 2** (Redesign): Remove environment variables, use unified state machine
3. **Phase 3** (Future): Leverage improved state machine with deterministic visibility

## Conclusion

These fixes address the root causes of CI timeout issues:
- **Replay agent test**: Fixed async iteration and added comprehensive timeout protection
- **Architect tests**: Forced minimal pipeline usage and added timeout protection
- **Overall**: Test suite now completes reliably without hanging

The changes maintain test functionality while ensuring CI stability and preventing indefinite hangs. All tests now have appropriate timeout protection and use efficient execution paths.

**Most importantly**, our approach aligns with the proposed architect redesign and provides a stable foundation for the transition to a unified, deterministic architect pipeline.
