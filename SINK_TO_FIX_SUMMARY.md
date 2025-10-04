# sink_to YAML Fix - Summary

**Date**: October 4, 2025  
**PR**: #501  
**Status**: ✅ **FIXED and Pushed**

---

## 🎯 The Issue (From Reviewer)

Your PR added `sink_to` field to `Step` class for persisting scalar values like counters, BUT it didn't work for YAML-loaded pipelines (the primary way users define pipelines).

**Example that failed**:
```yaml
- kind: step
  name: increment
  sink_to: counter  # ❌ This got IGNORED!
  agent: my_agent
```

The field was defined in the DSL but never wired through the blueprint loader, so YAML pipelines couldn't use it.

---

## 🔍 Root Cause

1. **`Step.from_callable()`** didn't have `sink_to` parameter
2. **Blueprint loader** (`loader.py`) never passed `model.sink_to` when creating Steps
3. **`executor_core.py`** didn't apply `sink_to` after successful execution
4. Result: Only worked for Python code, not YAML

---

## ✅ The Fix

### 1. Updated `Step.from_callable()` (step.py)
```python
def from_callable(
    ...
    sink_to: str | None = None,  # ← ADDED
    ...
)
```

### 2. Updated Blueprint Loader (loader.py)
Added `sink_to=model.sink_to` in **4 places**:
- Line 1702: `uses='agents.X'` path
- Line 1824: `agent` string import path  
- Line 1954: Registry-backed callables
- Line 1971: Plain `Step()` constructor

### 3. Added Execution Logic (executor_core.py, line 3415)
```python
# After successful step execution:
sink_path = getattr(step, "sink_to", None)
if sink_path and context is not None:
    set_nested_context_field(context, sink_path, output)
```

Uses `set_nested_context_field` to handle paths like `scratchpad.counter`.

---

## 🧪 Test Added

**`tests/integration/test_yaml_sink_to.py`**:
```python
async def test_sink_to_from_yaml():
    """Verify sink_to works for YAML-loaded pipelines."""
    yaml_content = """
steps:
  - kind: step
    name: increment
    agent: increment_agent
    sink_to: scratchpad.counter  # ✅ Now works!
    
  - kind: step
    name: check
    agent: check_agent
"""
    pipeline = Pipeline.from_yaml_text(yaml_content)
    result = await runner.run_async("5")
    
    # Verify counter was persisted
    assert result.final_pipeline_context.scratchpad["counter"] == 6
    assert result.output == "counter_is_6"
```

**Result**: ✅ Test passes!

---

## 📊 What Works Now

### ✅ YAML Pipelines
```yaml
steps:
  - kind: step
    name: work
    agent: my_agent
    sink_to: scratchpad.result  # ← Works!
```

### ✅ Python Code (already worked)
```python
Step(name="work", agent=agent, sink_to="scratchpad.result")
```

### ✅ Loops with sink_to
```yaml
- kind: loop
  body:
    - kind: step
      name: increment
      sink_to: scratchpad.counter  # ← Persists across iterations!
```

---

## 📝 Important Notes

### Path Syntax
- `sink_to: scratchpad.counter` - Stores in `context.scratchpad['counter']` ✅
- `sink_to: counter` - Stores as `context.counter` attribute (less common)

### Error Handling
- Sink failures never fail the step (wrapped in try/except)
- Warnings logged if sink path can't be set
- Falls back to `object.__setattr__()` for simple paths

---

## 🚀 Verification

### Tests Passing
```
tests/integration/test_yaml_sink_to.py .              [100%]
tests/integration/test_hitl_loop_minimal.py .         [ 20%]
tests/integration/test_hitl_loop_resume_simple.py .... [ 80%]

============================== 6 passed ===============================
```

### Commit
- **Hash**: `6bd81ef8`
- **Message**: "Fix(Blueprint): Wire sink_to through YAML blueprint loader"
- **Pushed**: ✅ Yes, to PR #501

---

## ✨ Summary

**Before**: `sink_to` field existed but was completely ignored for YAML pipelines  
**After**: `sink_to` works for both YAML and Python-defined pipelines  
**Impact**: Users can now persist scalar values (counters, flags, etc.) in YAML pipelines as documented

The reviewer's comment was **100% valid** and is now **completely fixed**! ✅

---

**Files Changed**:
- `flujo/domain/dsl/step.py` - Added `sink_to` parameter to `from_callable()`
- `flujo/domain/blueprint/loader.py` - Wire through `sink_to` in 4 places
- `flujo/application/core/executor_core.py` - Execute `sink_to` after success
- `tests/integration/test_yaml_sink_to.py` - Test YAML `sink_to` works (NEW)

**Status**: ✅ Ready for reviewer approval

