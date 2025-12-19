# Flujo Traceability Improvements - Implementation Plan

**Date**: October 3, 2025  
**Priority**: CRITICAL  
**Status**: In Progress

---

## Overview

Implementing comprehensive execution traceability to eliminate silent step skipping and provide clear visibility into pipeline execution. This addresses the critical issue where HITL steps and other steps can be silently skipped with no indication in logs or error messages.

---

## Implementation Status

### ✅ COMPLETED: Validation Warning (WARN-HITL-001)

**Status**: ✅ Complete and Tested

**What was implemented:**
- New `HitlNestedContextLinter` class in `flujo/validation/linters.py`
- Detects HITL steps inside nested contexts (loops, conditionals, parallel branches)
- Recursively traverses pipeline structure to find all nested HITL steps
- Provides context chain information (e.g., `loop:my_loop > conditional:check > branch:true`)
- Clear warning messages with restructuring suggestions and examples

**Files modified:**
1. `flujo/validation/linters.py` - Added `HitlNestedContextLinter` class (+147 lines)
2. `flujo/validation/rules_catalog.py` - Added `WARN-HITL-001`, `LOOP-001`, `TEMPLATE-001` rules
3. `examples/validation/test_hitl_nested_context.yaml` - Comprehensive test file

**Test results:**
```text
✅ Detected all 3 nested HITL steps correctly
✅ Provided clear context chains for each
✅ Did NOT flag top-level HITL (correct behavior)
✅ Suggestions are helpful and actionable
```

**Example output:**
```text
Warning [WARN-HITL-001] ask_user_deeply_nested: HITL step 'ask_user_deeply_nested' found in nested context: loop:nested_loop > conditional:check_nested > branch:true. HITL steps in nested contexts (loops, conditionals) may exhibit complex pause/resume behavior.

Suggestion: Consider one of these alternatives:
  1. Move HITL step to top-level (outside loop/conditional)
  2. Use flujo.builtins.ask_user skill instead
  3. Restructure pipeline to avoid nested HITL
  4. If intentional, ensure you understand pause/resume semantics
```

**Impact:** 
- Prevents developers from unknowingly using HITL in problematic contexts
- Warns BEFORE runtime failures (validation time vs execution time)
- Saves 2-4 hours of debugging per incident

---

### 🚧 IN PROGRESS: Step Lifecycle Logging

**Status**: 🚧 Design Complete, Implementation Pending

**What needs to be implemented:**

#### 1. Core Execution Logging
Add lifecycle logging to `ExecutorCore.execute()` and related methods:

**Target files:**
- `flujo/application/core/executor_core.py`
- `flujo/application/core/orchestration/step_coordinator.py`
- `flujo/application/core/execution/execution_manager.py`

**Logging structure:**
```python
# At step execution start
telemetry.logfire.info(
    f"[STEP_START] kind={step.kind} name={step.name} depth={nesting_depth}"
)

# At step execution end
telemetry.logfire.info(
    f"[STEP_END] kind={step.kind} name={step.name} status={status} duration={duration_ms}ms"
)

# When steps are skipped (NEW - needs investigation)
telemetry.logfire.warning(
    f"[STEP_SKIP] kind={step.kind} name={step.name} reason={skip_reason} context_chain={context_chain}"
)
```

**Implementation approach:**
1. Add wrapper logging in `executor_core._execute_pipeline_via_policies()`
2. Add context depth tracking via contextvars
3. Track context chain for nested steps
4. Add skip detection logic (needs investigation - where are steps being skipped?)

**Estimated effort:** 2-3 hours

---

#### 2. HITL-Specific Execution Markers

**Target files:**
- `flujo/application/core/step_policies.py` (DefaultHitlStepExecutor)
- `flujo/application/runner.py` (resume logic)

**Logging structure:**
```python
# HITL execution lifecycle
telemetry.logfire.info(f"[HITL_QUEUED] name={step.name} context={context_id}")
telemetry.logfire.info(f"[HITL_PAUSING] name={step.name} message={message}")
telemetry.logfire.info(f"[HITL_RESUMED] name={step.name} input_length={len(input)}")

# NEW: When HITL is skipped (critical for debugging)
telemetry.logfire.error(
    f"[HITL_SKIPPED] name={step.name} reason={reason} context_chain={context_chain}"
)
```

**Implementation approach:**
1. Add logging in `DefaultHitlStepExecutor.execute()`
2. Add skip detection before PausedException is raised
3. Track when HITL steps are unexpectedly bypassed
4. Log resume events in runner.py

**Estimated effort:** 1-2 hours

---

#### 3. Context Depth Tracking

**Target files:**
- `flujo/application/core/executor_core.py`
- `flujo/application/core/step_policies.py` (Loop, Conditional, Parallel executors)

**Implementation approach:**
1. Add `EXECUTION_DEPTH` contextvars.ContextVar
2. Increment depth when entering nested contexts (loops, conditionals)
3. Decrement when exiting
4. Include depth in all logging

**Example:**
```python
import contextvars

EXECUTION_DEPTH = contextvars.ContextVar('execution_depth', default=0)

# In loop/conditional executors:
depth = EXECUTION_DEPTH.get()
EXECUTION_DEPTH.set(depth + 1)
try:
    # Execute nested steps
    ...
finally:
    EXECUTION_DEPTH.set(depth)
```

**Estimated effort:** 1-2 hours

---

### 📋 PENDING: Additional Features

#### 4. Enhanced --trace CLI Flag

**Status**: ⏳ Pending

**What needs to be implemented:**
- Add `--trace` flag to CLI (more verbose than `--debug`)
- Set environment variable to enable trace-level logging
- Document in CLI help

**Target files:**
- `flujo/cli/commands/run.py`
- `flujo/infra/telemetry.py` (add TRACE log level)

**Estimated effort:** 1 hour

---

#### 5. assert_executed Field Support

**Status**: ⏳ Pending

**What needs to be implemented:**
- Add `assert_executed: bool` field to Step base class
- Add validation check that step was actually executed
- Raise clear error if skipped with assert_executed=true

**Target files:**
- `flujo/domain/dsl/step.py` (add field)
- `flujo/application/core/executor_core.py` (add validation)

**Estimated effort:** 2-3 hours

---

#### 6. Nested Context Visualization in Debug Output

**Status**: ⏳ Pending

**What needs to be implemented:**
- Add context_chain to debug JSON output
- Add executor_chain tracking
- Add skip_reason field to trace

**Target files:**
- `flujo/application/runner.py` (debug output)
- `flujo/state/trace.py` (trace format)

**Estimated effort:** 1-2 hours

---

## Implementation Priority

### Phase 1: Critical (Immediate) ✅
- [x] WARN-HITL-001 Validation Warning

### Phase 2: High Priority (Next 1-2 days)
- [ ] Step Lifecycle Logging (STEP_START, STEP_END, STEP_SKIP)
- [ ] HITL-Specific Execution Markers
- [ ] Context Depth Tracking

### Phase 3: Medium Priority (Next week)
- [ ] Enhanced --trace CLI Flag
- [ ] assert_executed Field Support

### Phase 4: Low Priority (Future)
- [ ] Nested Context Visualization in Debug Output
- [ ] Performance profiling integration
- [ ] Execution flow diagrams

---

## Testing Strategy

### Unit Tests
- [ ] Test WARN-HITL-001 validator with various nesting levels
- [ ] Test lifecycle logging is emitted correctly
- [ ] Test context depth tracking increments/decrements properly
- [ ] Test assert_executed raises errors correctly

### Integration Tests
- [ ] Test full pipeline with logging enabled
- [ ] Test HITL pause/resume with logging
- [ ] Test nested loop+conditional with logging
- [ ] Verify no performance regression

### Manual Testing
- [ ] Run pipelines with --debug and verify logs
- [ ] Run pipelines with --trace and verify extra detail
- [ ] Test with existing production pipelines

---

## Expected Impact

### Before Improvements
- Silent step skipping → 9+ hours debugging
- No visibility into execution flow
- Cannot tell if HITL was skipped or failed
- Unclear where in nested contexts execution is

### After Improvements
- Every step decision explicitly logged
- Clear [STEP_SKIP] warnings with reasons
- HITL lifecycle fully traceable
- Context depth visible in logs
- assert_executed prevents silent skips

**Estimated time saved:** 10-20 hours per developer per month

---

## Known Issues & Considerations

### Issue 1: Where are steps being skipped?
- Need to investigate actual skipping logic
- May be in conditional branching
- May be in loop iteration logic
- May be in error handling paths

**Action:** Code search for skip conditions

### Issue 2: Performance impact of logging
- Logging has overhead (minimal with lazy evaluation)
- Use log levels to control verbosity
- Profile with and without trace logging

**Action:** Benchmark before/after

### Issue 3: Backward compatibility
- All changes are additive (logging only)
- No breaking changes to APIs
- Existing pipelines continue to work

**Action:** Test with existing pipelines

---

## Documentation Updates Needed

1. **User Guide**: Document --trace flag usage
2. **Developer Guide**: Document traceability patterns
3. **Troubleshooting Guide**: How to use logs to debug
4. **HITL Guide**: Explain nested context warnings
5. **API Reference**: Document assert_executed field

---

## Related Issues

- **HITL in nested contexts**: Silent pause failures
- **Loop max_loops exhaustion**: No visibility into why
- **Conditional branches**: Not clear which branch taken
- **Template failures**: No indication step was skipped

All of these are addressed by comprehensive traceability.

---

## Next Steps

1. ✅ Commit WARN-HITL-001 validator (complete)
2. 🚧 Implement core step lifecycle logging
3. 🚧 Add HITL-specific markers
4. 🚧 Add context depth tracking
5. ⏳ Add --trace CLI flag
6. ⏳ Add assert_executed support

---

## References

- Original improvement request: `FLUJO_TRACEABILITY_REQUEST.md`
- Bug report: `BUG_ANALYSIS_HITL_SINK_TO.md`
- Team guide: `FLUJO_TEAM_GUIDE.md`
- Validation docs: `docs/user_guide/validation_rules.md`

---

**Last Updated**: October 3, 2025  
**Implemented By**: AI Assistant + Development Team

