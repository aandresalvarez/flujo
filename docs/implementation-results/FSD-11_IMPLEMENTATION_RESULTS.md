# FSD-11 Implementation Results: Signature-Aware Context Injection for Agent Wrappers

## Overview

This document summarizes the implementation and testing results for FSD-11, which addresses a critical bug in the Flujo framework where context injection was incorrectly based on the wrapper's signature rather than the underlying agent's signature.

## Problem Statement

The original FSD-11 implementation was incomplete. The framework was inspecting the `AsyncAgentWrapper`'s `run` method signature (which accepts `**kwargs`) instead of the underlying `pydantic-ai` agent's signature. This led to incorrect context injection decisions, causing `TypeError: run() got an unexpected keyword argument 'context'` errors.

## Root Cause Analysis

The issue was identified in two places:

1. **Step Logic (`flujo/application/core/executor_core.py`)**: The signature analysis was performed on the wrapper instead of the underlying agent
2. **AsyncAgentWrapper (`flujo/infra/agents.py`)**: The wrapper was not properly filtering context arguments before passing them to the underlying agent

## Implementation

### 1. Fixed Signature Analysis in Step Logic

**File**: `flujo/application/core/executor_core.py`

**Changes**:
- Added proper inspection of the underlying agent's signature for `AsyncAgentWrapper` instances
- The decision to inject context is now based on the ultimate target agent's signature

**Code**:
```python
# FR-35.1: Properly inspect the underlying agent's signature for AsyncAgentWrapper
# The decision to inject context must be based on the ultimate target agent's signature
target = getattr(current_agent, "_agent", current_agent)
func = getattr(target, "_step_callable", None)
if func is None:
    func = target.stream if stream and hasattr(target, "stream") else target.run
func = cast(Callable[..., Any], func)
spec = analyze_signature(func)
```

### 2. Enhanced Context Filtering in AsyncAgentWrapper

**File**: `flujo/infra/agents.py`

**Changes**:
- Implemented robust context filtering based on the underlying agent's signature
- Added proper error reporting with actual error type and message (FR-36)

**Code**:
```python
# FR-35.2: Filter kwargs before processing to avoid passing unwanted parameters
# This is the core fix for FSD-11 - only pass context if the underlying agent accepts it
from flujo.application.context_manager import _accepts_param

filtered_kwargs = {}
for key, value in kwargs.items():
    if key in ["context", "pipeline_context"]:
        # Only pass context if the underlying agent's run method accepts it
        accepts_context = _accepts_param(self._agent.run, "context")
        if accepts_context:
            filtered_kwargs[key] = value
        # Note: We don't pass context to the underlying agent if it doesn't accept it
        # This prevents the TypeError: run() got an unexpected keyword argument 'context'
    else:
        filtered_kwargs[key] = value
```

### 3. Fixed Signature Analysis for Never Type

**File**: `flujo/application/context_manager.py`

**Changes**:
- Enhanced `_accepts_param` function to properly handle `Never` type annotations
- Added support for both direct type comparison and string comparison for robustness

**Code**:
```python
# Check if the annotation is Never (either as type or string)
if p.annotation is Never or str(p.annotation) == "Never":
    result = False
else:
    result = True
```

## Testing Results

### Test Cases Implemented

1. **Test Case 1**: Stateless Agent (`make_agent_async`) - ✅ **FIXED**
2. **Test Case 2**: Context-Aware Agent (Explicit `context` Param) - ✅ **WORKS**
3. **Test Case 3**: Context-Aware Agent (`**kwargs`) - ✅ **WORKS**
4. **Test Case 4**: Error Propagation - ✅ **ENHANCED**
5. **Test Case 5**: Verify context is NOT passed to stateless agents - ✅ **FIXED**
6. **Test Case 6**: Context required but none provided - ✅ **WORKS**
7. **Test Case 7**: Signature analysis fix verification - ✅ **PASSES**
8. **Test Case 8**: Context filtering verification - ✅ **PASSES**

### Test Results

```
tests/integration/test_FSD11_bug_fix.py::test_fsd11_signature_analysis_fix PASSED
tests/integration/test_FSD11_bug_fix.py::test_fsd11_context_filtering_works PASSED
```

### Backward Compatibility

All existing context injection tests continue to pass:

```
tests/integration/test_context_injection.py::test_stateless_agent_no_context_injection PASSED
tests/integration/test_context_injection.py::test_stateless_agent_custom_no_context_injection PASSED
tests/integration/test_context_injection.py::test_context_aware_agent_explicit_context PASSED
tests/integration/test_context_injection.py::test_context_aware_agent_kwargs PASSED
tests/integration/test_context_injection.py::test_error_propagation PASSED
tests/integration/test_context_injection.py::test_backward_compatibility_existing_context_aware PASSED
tests/integration/test_context_injection.py::test_mixed_pipeline_stateless_and_context_aware PASSED
```

## Key Technical Insights

### 1. Signature Analysis Issue

The `pydantic-ai` agent's `run` method has the signature:
```python
def run(self, user_prompt: str, *, output_type: OutputSpec, ..., **_deprecated_kwargs: 'Never') -> AgentRunResult
```

The `**_deprecated_kwargs: 'Never'` parameter explicitly indicates that the method does NOT accept additional keyword arguments, but the original signature analysis was incorrectly interpreting this as accepting `**kwargs`.

### 2. Never Type Handling

The fix properly handles the `Never` type annotation by comparing both the type object and string representation:
```python
if p.annotation is Never or str(p.annotation) == "Never":
    result = False
```

### 3. Dual-Layer Protection

The implementation provides two layers of protection:
1. **Step Logic**: Correctly analyzes the underlying agent's signature
2. **AsyncAgentWrapper**: Filters context arguments before passing to underlying agent

## Impact

### Before Fix
- `make_agent_async` failed with `TypeError: run() got an unexpected keyword argument 'context'`
- Basic stateless agent usage was broken
- Framework behavior was unintuitive

### After Fix
- `make_agent_async` works correctly out-of-the-box
- Context injection is properly signature-aware
- Framework behavior is intuitive and reliable
- Enhanced error reporting provides better debugging information

## Conclusion

The FSD-11 implementation successfully addresses the critical bug by:

1. **Correcting signature analysis** to inspect the underlying agent's signature
2. **Implementing robust context filtering** in the AsyncAgentWrapper
3. **Enhancing error reporting** with proper error type and message information
4. **Maintaining backward compatibility** with existing context-aware agents

The fix ensures that the framework behaves as expected and provides a solid foundation for future development. The implementation follows the Single Responsibility Principle and provides comprehensive test coverage to prevent regressions.
