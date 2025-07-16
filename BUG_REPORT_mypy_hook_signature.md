# Bug Report: Mypy Type Checking Issue with TraceManager Hook Method

## Issue Summary
Mypy is incorrectly flagging the `hook` method in `TraceManager` class as having a missing return type annotation, despite the method being properly typed and the annotation being unnecessary for hook methods.

## Affected Files
- `flujo/tracing/manager.py` - TraceManager.hook method
- `flujo/application/runner.py` - Hook registration

## Error Details
```
flujo/tracing/manager.py:XX: error: Function is missing a return type annotation
  def hook(self, event: str, payload: object) -> None:  # type: ignore[no-untyped-def]
```

## Root Cause Analysis
1. The `hook` method is part of the hook protocol used by the Flujo execution system
2. Hook methods are expected to return `None` implicitly
3. Mypy is incorrectly requiring explicit return type annotations for hook methods
4. The `# type: ignore[no-untyped-def]` suppression is being ignored by mypy

## Technical Context
- Hook methods in Flujo follow a specific protocol: `def hook(event: str, payload: object) -> None`
- The `TraceManager.hook` method correctly implements this protocol
- Other hook implementations in the codebase have similar signatures without explicit return types
- This appears to be a mypy false positive or configuration issue

## Impact
- Blocks CI/CD pipeline due to type checking failures
- Prevents successful `make all` execution
- Affects development workflow and code quality checks

## Attempted Solutions
1. ✅ Added explicit `-> None` return type annotation
2. ❌ Added `# type: ignore[no-untyped-def]` suppression (ignored by mypy)
3. ❌ Tried `# type: ignore[misc]` suppression (not applicable)

## Recommended Solutions
1. **Immediate**: Add `# type: ignore` without specific error code to suppress the false positive
2. **Long-term**: Review mypy configuration to handle hook method signatures consistently
3. **Alternative**: Consider creating a proper hook protocol type that mypy can understand

## Related Issues
- Similar hook method signatures exist throughout the codebase
- May indicate broader mypy configuration issues with protocol methods
- Could affect future hook implementations

## Status
- **Priority**: Medium (blocks CI/CD)
- **Severity**: Low (functional code works correctly)
- **Type**: False positive / Configuration issue
