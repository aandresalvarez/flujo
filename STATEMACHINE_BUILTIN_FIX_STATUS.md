# StateMachine & Builtin Skills Consistency Fix - Status

## Branch
`feature/statemachine-builtin-consistency`

## Problem Statement
Builtin skills (`flujo.builtins.*`) had inconsistent parameter handling:
- Required `agent.params` syntax in most contexts
- Sometimes worked with `input:` in StateMachine but not elsewhere
- Poor error messages when parameters were missing

## Changes Implemented

### 1. Parameter Normalization (✅ DONE)
**File**: `flujo/application/core/step_policies.py`

Added `_normalize_builtin_params()` function (lines 2547-2602) that:
- Detects builtin skills by checking if `agent.id` starts with `"flujo.builtins."`
- Normalizes parameters from either `agent.params` or `step.input`
- Priority: `agent.params` > `step.input` > original data
- Called in `DefaultAgentStepExecutor.execute()` at line 1197

### 2. Integration Tests (✅ DONE)
**File**: `tests/integration/test_statemachine_builtins.py`

Created comprehensive test suite with 8 test cases:
- StateMachine with `params` and `input`
- Top-level steps with `params` and `input`
- Conditional branches
- `context_set` and `context_merge` builtins
- Dynamic state transitions

### 3. Current Issue (🔍 IN PROGRESS)
Tests show `context_merge` is being called but returning:
```json
{
  "merged_keys": [],
  "success": false
}
```

This suggests the normalization is working (params are being passed), but there's a secondary issue with how the agent_runner invokes builtin skills. The dict needs to be unpacked as kwargs.

## Next Steps

1. **Investigate agent_runner invocation** (HIGH PRIORITY)
   - Check how `agent_runner.run()` passes data to builtin skills
   - Ensure dict params are unpacked as kwargs: `skill(**params)` not `skill(params)`

2. **Fix test assertions**
   - Change `result.context` to `result.final_pipeline_context`
   - Update all 8 test cases

3. **Add validation** (NICE TO HAVE)
   - Add blueprint loader validation for missing builtin params
   - Provide clear error messages

4. **Update documentation**
   - Add examples showing both `params` and `input` work
   - Document the precedence order
   - Update `llm.md` and user guides

## Testing Strategy

Once the agent_runner issue is fixed, run:
```bash
# Our new tests
uv run python -m pytest tests/integration/test_statemachine_builtins.py -v

# Verify no regressions
make test-fast

# Full suite
make test
```

## Files Modified

- `flujo/application/core/step_policies.py` - Parameter normalization
- `tests/integration/test_statemachine_builtins.py` - New test suite (needs assertion fixes)

## Commit Message (Draft)

```
feat(builtins): normalize parameter handling for builtin skills

- Add _normalize_builtin_params() to support both 'agent.params' and 'input'
- Normalize parameter resolution across all step types
- Add comprehensive integration tests
- Fixes inconsistency where flujo.builtins.* only worked with specific syntax

BREAKING: None - this is additive, existing code still works
INCOMPLETE: agent_runner needs update to unpack dict params as kwargs

Related: User bug reports about context_merge failing outside StateMachine
```

## Additional Notes

- The core architectural fix (normalization) is sound and follows best practices
- The remaining issue is in the skill invocation layer, not the policy layer
- This is a non-breaking change - adds support for `input`, doesn't remove `params`

---

**Created**: 2025-10-03  
**Status**: In Progress - 70% complete  
**Blocker**: agent_runner dict unpacking

