# Testing Guidelines

## Core Principle: Preserve Test Integrity

**NEVER MODIFY TESTS UNLESS THERE IS A VERY GOOD REASON FOR IT**

Tests are the safety net that ensures code quality and prevents regressions. Modifying tests should be treated as a last resort, not a convenience.

## When Test Modification is Acceptable

### ✅ Valid Reasons to Modify Tests

1. **API Changes**: When the public API legitimately changes and tests need to reflect new interfaces
2. **Bug Fixes**: When tests were testing incorrect behavior and need to verify the fix
3. **Test Infrastructure**: Updating test utilities, fixtures, or shared testing code
4. **Performance**: Converting slow tests to use mocks or improving test execution time
5. **Flaky Tests**: Fixing tests that are unreliable due to timing issues or race conditions

### ❌ Invalid Reasons to Modify Tests

1. **Making tests pass**: Changing assertions to match broken code
2. **Convenience**: Modifying tests because they're "in the way" of new features
3. **Shortcuts**: Removing test coverage to speed up development
4. **Misunderstanding**: Changing tests without understanding what they're validating

## Best Practices

### Before Modifying Any Test

1. **Understand the intent**: What behavior is the test validating?
2. **Verify the failure**: Is the test failing because of a real issue?
3. **Consider alternatives**: Can you fix the code instead of the test?
4. **Document changes**: Clearly explain why the test modification was necessary

### Test Modification Process

1. **Isolate changes**: Modify only what's absolutely necessary
2. **Maintain coverage**: Ensure test coverage doesn't decrease
3. **Preserve intent**: Keep the original validation purpose intact
4. **Add context**: Include comments explaining the modification

### Flujo-Specific Test Considerations

- **Markers**: Respect test markers (`fast`, `slow`, `serial`, `benchmark`)
- **Parallel execution**: Don't break tests that run in parallel
- **State isolation**: Ensure tests don't interfere with each other
- **Performance**: Be mindful of the 2400+ test suite performance

## Red Flags

If you find yourself wanting to modify tests for these reasons, stop and reconsider:

- "This test is blocking my feature"
- "The test is too strict"
- "I'll just comment this out temporarily"
- "The test doesn't understand my new approach"

## Alternative Approaches

Instead of modifying tests, consider:

- **Refactoring code** to match existing contracts
- **Adding new tests** alongside existing ones
- **Using feature flags** to maintain backward compatibility
- **Gradual migration** with both old and new tests during transition