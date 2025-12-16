# Deprecation Migration Guide

This guide documents deprecated patterns in Flujo and their recommended replacements.

---

## Table of Contents

1. [Exception Changes](#exception-changes)
2. [OptimizationConfig Removal](#optimizationconfig-removal)
3. [Global Agent Instances](#global-agent-instances)
4. [Scratchpad Migration](#scratchpad-migration)

---

## Exception Changes

### `ImproperStepInvocationError` → `StepInvocationError`

**Deprecated in:** v0.4.0  
**Will be removed in:** v1.0.0

**Before:**
```python
from flujo.exceptions import ImproperStepInvocationError

try:
    step("input")  # Direct invocation
except ImproperStepInvocationError:
    print("Step cannot be called directly")
```

**After:**
```python
from flujo.exceptions import StepInvocationError

try:
    step.run("input")  # Improper access
except StepInvocationError:
    print("Step cannot be called directly")
```

**Migration notes:**
- `StepInvocationError` provides enhanced error messages with suggestions
- Both exceptions inherit from `ExecutionError`
- The new exception includes structured fields: `suggestion` and `code`

---

## OptimizationConfig Removal

**Deprecated in:** v0.4.39  
**Will be removed in:** v1.0.0

The optimization layer has been completely removed from Flujo. `OptimizationConfig` now does nothing.

**Before:**
```python
from flujo.application.core.executor_core import ExecutorCore, OptimizationConfig

config = OptimizationConfig(
    enable_object_pool=True,
    enable_memory_optimization=True,
)
executor = ExecutorCore(optimization_config=config)
```

**After:**
```python
from flujo.application.core.executor_core import ExecutorCore

# Simply remove OptimizationConfig - it has no effect
executor = ExecutorCore()
```

**Migration notes:**
- All `OptimizationConfig` parameters are ignored
- Instantiating `OptimizationConfig` emits a `DeprecationWarning`
- Remove any imports and usages of `OptimizationConfig`

---

## Global Agent Instances

**Deprecated in:** v0.5.0  
**Will be removed in:** v1.0.0

Global agent instances have been replaced with factory functions.

**Deprecated globals:**
- `review_agent`
- `solution_agent`
- `validator_agent`
- `reflection_agent`
- `self_improvement_agent`
- `repair_agent`

**Before:**
```python
from flujo.agents import review_agent

result = await review_agent.run(data)  # Raises AttributeError
```

**After:**
```python
from flujo.agents import make_review_agent

# Create a fresh agent instance
agent = make_review_agent()
result = await agent.run(data)
```

**Available factories:**
| Deprecated Global | Replacement Factory |
|------------------|---------------------|
| `review_agent` | `make_review_agent()` |
| `solution_agent` | `make_solution_agent()` |
| `validator_agent` | `make_validator_agent()` |
| `reflection_agent` | `get_reflection_agent()` |
| `self_improvement_agent` | `make_self_improvement_agent()` |

---

## Scratchpad Migration

**Deprecated in:** v0.5.0  
**Removed in:** v0.6.0

The `scratchpad` field has been completely removed from `PipelineContext`.

**Before:**
```python
context.scratchpad["status"] = "running"
context.scratchpad["step_data"] = {"key": "value"}
```

**After:**
```python
# Use typed context fields instead
context.status = "running"
context.step_outputs["step_name"] = {"key": "value"}
```

**Typed field mappings:**

| Old `scratchpad` key | New typed field |
|---------------------|-----------------|
| `scratchpad["status"]` | `context.status` |
| `scratchpad["steps"]` | `context.step_outputs` |
| `scratchpad["current_state"]` | `context.current_state` |
| `scratchpad["pause_message"]` | `context.pause_message` |
| `scratchpad["initial_input"]` | `context.import_artifacts["initial_input"]` |

**Migration notes:**
- Any payload containing `scratchpad` key will be rejected at validation
- See `docs/guides/scratchpad_migration.md` for complete migration guidance
- Use `sink_to` on steps to store scalar values to context paths

---

## Checking for Deprecation Warnings

Enable warnings to detect deprecated usage in your code:

```bash
python -W default::DeprecationWarning your_script.py
```

Or in your test configuration:

```python
# pytest.ini or pyproject.toml
filterwarnings = ["error::DeprecationWarning"]
```
