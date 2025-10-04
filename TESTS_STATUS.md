# Test Status for HITL Loop Fix

**Date**: October 4, 2025

## Summary

I've created comprehensive regression tests for the HITL in loops bug, and implemented a fix in `step_policies.py`. However, **the tests are currently failing**, which means **the fix needs further work**.

## What I've Implemented

### 1. Fix in `step_policies.py` (DefaultLoopStepExecutor)
- **Lines 4479-4527**: Resume detection logic
- **Lines 4746-4782**: Data routing on resume  
- **Lines 4856-4858**: State persistence on pause
- **Lines 5765-5784**: Cleanup on completion

### 2. Test Files Created
- `tests/integration/test_hitl_loop_resume_simple.py` - Simplified regression tests
- `tests/integration/test_hitl_loop_minimal.py` - Minimal reproduction test

## Current Status: Tests FAILING ❌

The minimal test (`test_hitl_loop_minimal.py`) shows:
- ✅ HITL **does pause** correctly
- ❌ On resume, loop **still creates nested instance** (visible in trace)
- ❌ `step_history` is empty because nested loop doesn't complete

### Evidence from Test Output:
```
Span(name='simple_loop', ...
  children=[
    Span(name='simple_loop', ...  # NESTED LOOP - BUG STILL EXISTS!
```

## Why the Fix Isn't Working Yet

Possible issues with my implementation:

### Issue 1: Resume Detection Not Triggering
My code checks:
```python
if (maybe_iteration >= 1 and maybe_index >= 0 and maybe_status == "paused"):
    is_resuming = True
```

But the status might be getting cleared or changed before the loop policy is invoked on resume.

### Issue 2: Wrong Place to Check Status
The runner might be calling the loop policy with status already changed to "running".

### Issue 3: `.get()` vs `.pop()`
I changed from `.pop()` to `.get()` to preserve the state, but this might interfere with the existing cleanup logic.

## Next Steps

### Option 1: Debug the Fix
1. Add more telemetry logging to see when resume detection triggers
2. Check what the actual status is when loop policy is called on resume
3. Verify `loop_last_output` is being saved correctly

### Option 2: Different Approach
Instead of detecting resume in the loop policy, modify the **runner** to pass a resume flag to the loop, or handle loop resume at a higher level.

### Option 3: Simpler Fix
The core issue is that on resume, the runner passes human input as loop data. Maybe the fix should be in the **runner**, not the loop policy:
- Runner should detect if it's resuming into a loop
- Pass a special parameter or use a different code path
- Don't re-execute the loop step, but call a `resume_loop()` method instead

## Recommendation

The **architectural issue** is that loops are **stateful/complex** but the runner treats them as **atomic/restartable**. 

The cleanest fix would be:
1. Add a `resume()` method to LoopStep
2. Runner detects loop resume scenario
3. Calls `loop.resume(human_input)` instead of `loop.execute(human_input)`
4. Resume method continues from saved position

This requires changes to:
- `LoopStep` class (add resume method)
- `DefaultLoopStepExecutor` (implement resume logic)
- `Runner` (detect loop resume, call appropriate method)

## User's Next Steps

You have several options:

1. **Continue debugging my fix**: Add logging to see why resume detection isn't working
2. **Try architectural fix**: Implement proper resume protocol for loops
3. **Report to Flujo team**: Share the test that reproduces the bug and let them implement the fix
4. **Use workaround**: Avoid HITL inside loops for now (flatten the structure)

The tests I created **will be valuable** regardless - they clearly demonstrate the bug and will verify when it's fixed!

---

**Status**: 🔴 **Tests created but failing - fix needs more work**  
**Impact**: **Still blocks HITL in loops use case**  
**Tests**: Ready and waiting for working fix

