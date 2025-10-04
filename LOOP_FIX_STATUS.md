# Loop Fix Status

## Current Issue

The `DefaultLoopStepExecutor` in `flujo/application/core/step_policies.py` has a critical bug where it sets `loop_body_steps = None` for single-step loops (line 4503) but then unconditionally tries to iterate over it (line 4729), causing:

```
TypeError: object of type 'NoneType' has no len()
```

## Failing Tests

1. `tests/integration/test_caching_and_fallbacks.py::test_loop_step_fallback_continues`
2. `tests/application/core/test_executor_core_loop_step_migration.py::TestLoopStepMigration::test_handle_loop_step_body_step_failures`
3. `tests/integration/test_hitl_integration.py::test_map_with_hitl_pauses_each_item_and_collects_results`

## Root Cause

Lines 4499-4503:
```python
if hasattr(body_pipeline, "steps") and body_pipeline.steps and len(body_pipeline.steps) > 1:
    loop_body_steps = body_pipeline.steps
else:
    # For single step, use regular execution to preserve fallback behavior
    loop_body_steps = None
```

Line 4729:
```python
for step_idx in range(current_step_index, len(loop_body_steps)):  # ← Fails when None!
```

## Required Fix

Add conditional execution starting at line 4715:

```python
# CRITICAL FIX: Choose execution method based on step count
if loop_body_steps is not None:
    # Step-by-step execution for multiple steps (handles HITL pauses)
    original_cache_enabled = getattr(core, "_enable_cache", True)
    pipeline_result = None
    
    try:
        core._enable_cache = False
        # ... (existing step-by-step execution code, lines 4722-4831, indented by 4 spaces)
    finally:
        core._enable_cache = original_cache_enabled
else:
    # Regular execution for single step (preserves fallback behavior)
    pipeline_result = await core._execute_pipeline_via_policies(
        body_pipeline,
        current_data,
        iteration_context,
        resources,
        limits,
        breach_event,
    )
```

## Implementation Challenge

The fix requires indenting lines 4722-4831 (~110 lines) by 4 spaces to nest them inside the `if loop_body_steps is not None:` block. This is tedious and error-prone to do line-by-line with search-replace.

## Recommended Approach

Use a text editor or script to:
1. Insert `if loop_body_steps is not None:` at line 4716
2. Indent lines 4717-4834 by 4 spaces
3. Add the `else:` clause with regular execution after line 4834
4. Test compilation and run the failing tests

## Alternative: Remove Conditional

If step-by-step execution works for single-step loops, we could simplify by removing the conditional:

```python
# Always use step-by-step execution
if hasattr(body_pipeline, "steps") and body_pipeline.steps:
    loop_body_steps = body_pipeline.steps
else:
    # Convert single step to list
    loop_body_steps = [body_pipeline]
```

This would require testing to ensure fallback behavior still works correctly.

