# HITL in Loops - Regression Test Suite

**Purpose**: Ensure the HITL (Human-In-The-Loop) in loops bug never regresses.

**Bug Fixed**: PR #500 was incomplete. Loops would create nested instances on resume instead of continuing from saved position, causing infinite nesting and preventing agent outputs from being captured.

---

## Test Files

### `test_hitl_loop_resume_fix.py`
Comprehensive integration tests covering all aspects of the fix:

1. **`test_hitl_in_loop_no_nesting_on_resume`** (CRITICAL)
   - **What it tests**: Loop doesn't create nested instances on resume
   - **How it catches the bug**: Counts loop steps in result, verifies exactly 1 (not 3+)
   - **Why it's important**: This was the primary symptom users reported

2. **`test_hitl_in_loop_agent_output_captured`** (CRITICAL)
   - **What it tests**: Agent outputs are captured before HITL pause
   - **How it catches the bug**: Tracks execution order, verifies agent completes
   - **Why it's important**: Missing `agent.output` events were key evidence of the bug

3. **`test_hitl_in_loop_data_flow_integrity`** (CRITICAL)
   - **What it tests**: Data flows correctly across pause/resume
   - **How it catches the bug**: Verifies human input goes to HITL step, not loop
   - **Why it's important**: Wrong data routing caused the nesting

4. **`test_hitl_in_loop_multiple_iterations`** (CRITICAL)
   - **What it tests**: Loops can complete multiple iterations with HITL
   - **How it catches the bug**: Tracks iteration numbers, verifies [1,2,3] not [1,1,1]
   - **Why it's important**: Proves loop progresses instead of restarting

5. **`test_hitl_in_loop_cleanup_on_completion`**
   - **What it tests**: Loop state is cleaned up on completion
   - **How it catches the bug**: Verifies resume state is cleared
   - **Why it's important**: Prevents state pollution between runs

6. **`test_hitl_in_loop_resume_at_correct_step`**
   - **What it tests**: Resume continues from exact saved position
   - **How it catches the bug**: Tracks step execution order
   - **Why it's important**: Verifies step-by-step execution works

---

## Running the Tests

### Quick Run (All HITL Loop Tests)
```bash
cd /Users/alvaro1/Documents/Coral/Code/flujo/flujo

# Run all HITL loop tests
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py \
  --timeout 180 \
  --tb

# Expected: All 6 tests PASS ✅
```

### Run Individual Critical Tests
```bash
# Test 1: No nesting
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_no_nesting_on_resume

# Test 2: Agent outputs captured
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_agent_output_captured

# Test 3: Data flow integrity
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_data_flow_integrity
```

### Run in CI/CD
```bash
# These tests are marked @pytest.mark.slow and @pytest.mark.serial
# Include them in the slow test phase:

.venv/bin/python scripts/run_targeted_tests.py \
  --full-suite \
  --markers "slow and not benchmark" \
  --workers 1 \
  --timeout 180
```

---

## Test Strategy & Markers

### Markers Used
- `@pytest.mark.slow`: These tests involve HITL state management (60-180s)
- `@pytest.mark.serial`: Avoid SQLite contention from parallel execution
- `@pytest.mark.timeout(N)`: Explicit timeouts (120-180s per test)

### Why Slow/Serial?
- HITL steps use SQLite backend for state persistence
- Pause/resume cycles need sequential execution
- State management needs time to persist/restore

### Test Isolation
Each test:
- Creates a fresh pipeline
- Uses isolated context
- Cleans up state after completion
- Can run independently or as a suite

---

## Expected Results

### All Tests Pass ✅
```
test_hitl_in_loop_no_nesting_on_resume PASSED                    [16%]
test_hitl_in_loop_agent_output_captured PASSED                   [33%]
test_hitl_in_loop_data_flow_integrity PASSED                     [50%]
test_hitl_in_loop_multiple_iterations PASSED                     [66%]
test_hitl_in_loop_cleanup_on_completion PASSED                   [83%]
test_hitl_in_loop_resume_at_correct_step PASSED                  [100%]

====== 6 passed in 360.00s ======
```

### If a Test Fails 🔴

**Test 1 Fails (No Nesting):**
- **Symptom**: "Expected 1 loop step result, got 3"
- **Root cause**: Loops are nesting on resume again
- **Check**: Resume detection logic in `DefaultLoopStepExecutor`

**Test 2 Fails (Agent Outputs):**
- **Symptom**: "Agent step should execute before HITL pause"
- **Root cause**: Agent step restarting instead of completing
- **Check**: Step-by-step execution order

**Test 3 Fails (Data Flow):**
- **Symptom**: "Step should receive human input. Got: processed_initial_data"
- **Root cause**: Human input not routed to correct step
- **Check**: Data routing logic on resume (lines 4771-4782)

**Test 4 Fails (Multiple Iterations):**
- **Symptom**: "Iterations should be [1,2,3]. Got [1,1,1]"
- **Root cause**: Loop restarting instead of incrementing
- **Check**: Iteration counter logic

**Test 5 Fails (Cleanup):**
- **Symptom**: "loop_iteration should be cleared"
- **Root cause**: Resume state not cleaned up
- **Check**: Cleanup logic (lines 5765-5784)

**Test 6 Fails (Resume Position):**
- **Symptom**: "step0 should execute exactly once. Got 2 times"
- **Root cause**: Loop restarting from step 0 on resume
- **Check**: Resume position restoration (lines 4479-4527)

---

## Integration with CI/CD

### Pre-Commit Hook
```bash
# Add to .git/hooks/pre-commit
.venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_no_nesting_on_resume \
  --timeout 120
```

### CI Pipeline (GitHub Actions)
```yaml
- name: Run HITL Loop Regression Tests
  run: |
    .venv/bin/python scripts/run_targeted_tests.py \
      tests/integration/test_hitl_loop_resume_fix.py \
      --timeout 180 \
      --fail-fast
  
- name: Upload Test Results
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: hitl-loop-test-failures
    path: output/failure_summary_*.txt
```

### Regression Alert
If these tests fail in CI:
1. **STOP THE MERGE** - This is a critical regression
2. Review recent changes to `step_policies.py` (DefaultLoopStepExecutor)
3. Check if pause/resume logic was modified
4. Verify no changes to runner's resume handling
5. Run full test suite to check for related failures

---

## Maintenance

### When to Update These Tests

**Add new tests when:**
- New HITL features added (e.g., conditional HITL, parallel HITL)
- Loop policy changes significantly
- New resume scenarios discovered
- User reports similar nesting issues

**Update existing tests when:**
- Pipeline DSL syntax changes
- Context structure changes
- Test becomes flaky (increase timeout, add retries)

### Test Review Checklist

When reviewing changes to loop policy:
- [ ] All 6 tests still pass
- [ ] No new timeouts or flakiness
- [ ] Test execution time hasn't significantly increased
- [ ] Tests still catch the original bug (manually verify by reverting fix)
- [ ] New edge cases are covered

---

## Debugging Failed Tests

### Enable Verbose Logging
```bash
FLUJO_DEBUG=1 .venv/bin/python scripts/run_targeted_tests.py \
  tests/integration/test_hitl_loop_resume_fix.py::test_hitl_in_loop_no_nesting_on_resume \
  --tb \
  -v
```

### Check Trace Structure
```python
# Add to test to dump trace
import json
print(json.dumps(result.to_dict(), indent=2))
```

### Verify Context State
```python
# Add to test to inspect context
print("Context scratchpad:", result.final_pipeline_context.scratchpad)
print("Saved state:", {
    "loop_iteration": ctx.scratchpad.get("loop_iteration"),
    "loop_step_index": ctx.scratchpad.get("loop_step_index"),
    "loop_last_output": ctx.scratchpad.get("loop_last_output"),
})
```

### Compare with Working Trace
Save a known-good trace and compare:
```bash
# Known good
FLUJO_DEBUG=1 .venv/bin/python scripts/run_targeted_tests.py ... > good_trace.log

# Current (possibly broken)
FLUJO_DEBUG=1 .venv/bin/python scripts/run_targeted_tests.py ... > bad_trace.log

# Compare
diff good_trace.log bad_trace.log
```

---

## Performance Benchmarks

### Expected Test Times
- Test 1 (No Nesting): ~30-60s
- Test 2 (Agent Outputs): ~30-60s
- Test 3 (Data Flow): ~30-60s
- Test 4 (Multiple Iterations): ~60-120s (multiple pause/resume cycles)
- Test 5 (Cleanup): ~30-60s
- Test 6 (Resume Position): ~30-60s

**Total Suite**: ~3-6 minutes (serial execution)

### Optimization Notes
- Tests use `flujo.builtins.passthrough` (no actual LLM calls)
- No network I/O required
- SQLite operations are local filesystem
- Timeouts are conservative (can be reduced if stable)

---

## Related Documentation

- **Fix Documentation**: `HITL_LOOP_FIX_EXPLANATION.md`
- **Bug Analysis**: `PR_500_BUG_ANALYSIS.md`
- **Test Guide**: `scripts/test_guide.md`
- **Team Guide**: `FLUJO_TEAM_GUIDE.md` (Principle #3: Context Idempotency)

---

**Status**: ✅ Complete test coverage for HITL in loops bug  
**Maintenance**: Review quarterly, update when loop policy changes  
**Priority**: P0 - These tests are critical regression guards

