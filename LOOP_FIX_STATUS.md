# Loop Fix Status - UPDATED

## ✅ Fixed: TypeError for Single-Step Loops

**Status**: RESOLVED in commit `5ba824a5`

The `TypeError: object of type 'NoneType' has no len()` issue has been fixed by:
1. Always initializing `loop_body_steps` as a list (empty for single-step pipelines)
2. Adding conditional execution: `if len(loop_body_steps) > 0` for multi-step, `else` for single-step
3. Single-step loops now use regular `core._execute_pipeline_via_policies` execution

**Tests Now Passing**:
- ✅ `tests/integration/test_caching_and_fallbacks.py::test_loop_step_fallback_continues`
- ✅ `tests/application/core/test_executor_core_loop_step_migration.py::TestLoopStepMigration::test_handle_loop_step_body_step_failures`

## ❌ Remaining Issue: HITL Pause/Resume in Loops

**Status**: IN PROGRESS

**Failing Test**:
- `tests/integration/test_hitl_integration.py::test_map_with_hitl_pauses_each_item_and_collects_results`

**Problem**: 
After HITL pause in iteration 1 and resume, the loop completes iteration 2 successfully but then tries to start iteration 3, exceeding `max_loops=2` and causing a "failed" status instead of "paused".

**Root Cause**:
Lines 4806-4807 in `step_policies.py`:
```python
current_context.scratchpad["loop_step_index"] = 0  # Start from beginning of next iteration
current_context.scratchpad["loop_iteration"] = iteration_count + 1  # Next iteration
```

This is incorrectly setting the loop to resume at `iteration_count + 1` (iteration 2) when it should resume at the current iteration (iteration 1) but skip to the next step after the HITL pause.

**Expected Behavior**:
1. Loop starts iteration 1, pauses at step 2 (AnnotateItem - HITL)
2. User provides input
3. Loop resumes iteration 1 at step 3 (CombineItemAndNote)
4. Iteration 1 completes
5. Loop starts iteration 2
6. Iteration 2 completes
7. Loop exits successfully with status="completed" (not "failed")

**Current Behavior**:
1. Loop starts iteration 1, pauses at step 2
2. User provides input
3. Loop starts iteration 2 (skipping rest of iteration 1!)
4. Iteration 2 completes
5. Loop tries to start iteration 3, exceeds max_loops
6. Loop fails with status="failed"

**Required Fix**:
Change lines 4806-4807 to resume at the NEXT step of the CURRENT iteration:
```python
current_context.scratchpad["loop_step_index"] = step_idx + 1  # Next step in current iteration
current_context.scratchpad["loop_iteration"] = iteration_count  # Same iteration
```

This will ensure:
- HITL pause/resume continues from the next step in the same iteration
- The loop doesn't skip steps or iterations
- The loop respects `max_loops` correctly
