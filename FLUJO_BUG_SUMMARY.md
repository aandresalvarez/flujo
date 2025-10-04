# 🚨 CRITICAL: Flujo HITL Loop Execution Bug

## **The Problem**

PR #499 only fixed **validation** but not the **underlying execution bug**. When HITL steps pause inside loops, Flujo **restarts the entire loop iteration** instead of continuing from where it paused.

## **Evidence**

**Debug trace shows**:
- Loop always stuck at `iteration: 1` 
- Steps after HITL never execute
- Creates nested loops instead of new iterations
- Context updates are lost

## **Root Cause**

**File**: `flujo/application/core/step_policies.py:4474-4774`

```python
for iteration_count in range(1, max_loops + 1):
    try:
        # Execute loop body
        pipeline_result = await core._execute_pipeline_via_policies(...)
    except PausedException as e:
        raise e  # ← BUG: Restarts entire loop!
```

**The bug**: `PausedException` re-raise causes **entire loop to restart** instead of continuing from paused position.

## **Impact**

**ALL interactive loop patterns are broken**:
- ❌ Clarification workflows
- ❌ Approval workflows  
- ❌ Multi-turn conversations
- ❌ Progressive refinement
- ❌ Any loop with HITL

## **What Needs Fixing**

Flujo needs to:
1. Track step position within loop body
2. Save position when HITL pauses
3. Resume from next step (not loop start)
4. Only increment iteration after full body completes

## **Status**

🚨 **CRITICAL BUG CONFIRMED**  
💥 **ALL INTERACTIVE LOOPS BROKEN**  
🔥 **IMMEDIATE ATTENTION REQUIRED**

**This is a core execution engine bug, not a configuration issue.**
