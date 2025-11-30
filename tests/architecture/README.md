# Architecture Compliance Tests

This directory contains automated tests that enforce the architectural standards and type safety requirements established in `FLUJO_TEAM_GUIDE.md` and `docs/development/type_safety.md`.

## Overview

These tests serve as **quality gates** that must pass before any PR can be merged. They ensure that:

1. **Type Safety**: No new `Any` types are introduced, proper type annotations are used
2. **Architecture Compliance**: Code follows policy-driven architecture patterns
3. **Code Quality**: All quality standards are maintained
4. **CI/CD Integration**: Build pipeline properly enforces quality gates

## Test Files

### `test_type_safety_compliance.py`
Tests for type safety compliance:

- **Any Type Prevention**: Detects new `Any` type usages beyond baseline
- **JSONObject Usage**: Ensures `JSONObject` is used instead of `Dict[str, Any]`
- **Typed Fixtures**: Verifies test files use typed fixtures from `tests/test_types/`
- **Quality Gates**: Runs `make all` and ensures it passes

### `test_architecture_compliance.py`
Tests for architectural pattern compliance:

- **Policy-Driven Architecture**: Ensures step logic stays in policies, not `ExecutorCore`
- **Exception Handling**: Verifies control flow exceptions are re-raised properly
- **Context Isolation**: Checks that complex steps use `ContextManager.isolate()`
- **Quota Patterns**: Ensures resource usage follows Reserve→Execute→Reconcile
- **Fallback Logic**: Verifies infinite loop detection in fallback chains
- **Type Annotations**: Checks all functions have complete type annotations

### `test_ci_configuration.py`
Tests for CI/CD and development workflow compliance:

- **GitHub Actions**: Verifies CI runs architecture tests
- **Makefile**: Checks for required quality targets
- **Pre-commit Hooks**: Ensures mypy and ruff hooks are configured
- **VS Code Settings**: Verifies IDE enforces quality standards
- **Development Workflow**: Checks scripts and documentation completeness

## Running the Tests

### Run All Architecture Tests
```bash
# From project root
pytest tests/architecture/ -v
```

### Run Specific Test Categories
```bash
# Type safety only
pytest tests/architecture/test_type_safety_compliance.py -v

# Architecture compliance only
pytest tests/architecture/test_architecture_compliance.py -v

# CI configuration only
pytest tests/architecture/test_ci_configuration.py -v
```

### Run as Part of Quality Gates
```bash
# Run all quality checks (includes architecture tests)
make all

# Or run architecture tests specifically
make test-architecture  # (if added to Makefile)
```

## Test Failure Behavior

These tests are designed to **fail builds** when violations are detected:

### Type Safety Violations
- New `Any` types introduced → Test fails
- `Dict[str, Any]` used instead of `JSONObject` → Test fails
- Missing type annotations → Test fails (with baseline allowance)

### Architecture Violations
- Step logic in `ExecutorCore` → Test fails
- Control flow exceptions converted to failures → Test fails
- Missing context isolation → Test fails

### CI/CD Violations
- Missing architecture tests in CI → Test fails
- Missing quality targets in Makefile → Test fails
- Missing pre-commit hooks → Test fails

## Baseline Management

Some tests use **baselines** to allow gradual improvement:

### Any Type Baseline
- Current baseline: ~1,627 `Any` occurrences
- Test allows some growth but prevents significant new usage
- Should be reduced over time as code is migrated

### Type Annotation Baseline
- Current baseline: 100 functions without complete annotations
- Test prevents significant regressions
- Should be reduced to 0 over time

## Integration with CI/CD

### GitHub Actions
```yaml
# .github/workflows/ci.yml
- name: Run Architecture Compliance Tests
  run: |
    pytest tests/architecture/ -v --tb=short
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: architecture-tests
      name: Architecture Compliance
      entry: pytest tests/architecture/ --tb=line
      language: system
      pass_filenames: false
```

### Makefile Integration
```makefile
# Makefile
test-architecture:
    pytest tests/architecture/ -v

all: typecheck lint test test-architecture
```

## Development Workflow

### For Contributors
1. **Before committing**: Run `make all` (includes architecture tests)
2. **During development**: Run architecture tests frequently
3. **When adding features**: Ensure new code passes all architecture tests

### For Maintainers
1. **Review PRs**: Check that architecture tests pass
2. **Merge gate**: Block merges if architecture tests fail
3. **Regular audits**: Run full architecture test suite weekly

## Troubleshooting

### Common Test Failures

#### "Found new Any type usages"
**Solution**: Replace `Any` with specific types or use generics. Document if `Any` is truly necessary.

#### "Dict[str, Any] used instead of JSONObject"
**Solution**: Import `JSONObject` from `flujo.type_definitions.common` and use it instead.

#### "Step logic found in ExecutorCore"
**Solution**: Move step-specific logic to appropriate policy classes in `step_policies.py`.

#### "Control flow exception converted to failure"
**Solution**: Re-raise `PausedException`, `PipelineAbortSignal`, `InfiniteRedirectError` instead of returning `StepResult`.

## Related Documentation

- [`FLUJO_TEAM_GUIDE.md`](../FLUJO_TEAM_GUIDE.md) - Complete architectural standards
- [`docs/development/type_safety.md`](../../docs/development/type_safety.md) - Type safety patterns
- [`flujo/type_definitions/`](../../flujo/type_definitions/) - Type definitions
- [`tests/test_types/`](../test_types/) - Typed test utilities

## Contributing

When adding new architecture tests:

1. Add tests to appropriate file based on category
2. Include clear failure messages explaining violations
3. Update this README if adding new test categories
4. Ensure tests can run in CI/CD environment
5. Add documentation for any new baselines or thresholds
