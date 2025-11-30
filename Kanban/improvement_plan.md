# Flujo Architectural Improvement Plan

> **Alignment**: All recommendations are validated against `FLUJO_TEAM_GUIDE.md` to ensure consistency with established architectural principles.

## Executive Summary

This plan outlines strategic architectural improvements for Flujo, prioritized by:
1. **Guide Alignment**: How well the change aligns with FLUJO_TEAM_GUIDE.md
2. **Impact**: Business value and developer productivity gains
3. **Risk**: Breaking change potential and regression risk
4. **Effort**: Development time and complexity

---

## Phase 1: Fully Aligned Improvements (Immediate Priority)

### 1.1 Modularization of Remaining Monolithic Components

- **due**: 2025-12-15
- **tags**: [architecture, monolith, refactor]
- **priority**: high
- **workload**: Medium
- **guide_alignment**: ⭐⭐⭐ Perfect

#### Justification

**FLUJO_TEAM_GUIDE.md Reference**:
> "Module Decomposition Completed: The legacy monolithic module `flujo/application/core/ultra_executor.py` has been decomposed."

The guide explicitly endorses modular decomposition. Continuing this pattern for remaining large files follows established precedent.

#### Current State

Based on Phase 3 and fix_items.md, these decompositions are complete:
- ✅ `ultra_executor.py` → executor_core.py/protocols/default_components
- ✅ `agent_orchestrator.py` → agent_execution_runner.py/agent_plugin_runner.py
- ✅ `runner.py` → runner_execution.py/runner_telemetry.py
- ✅ `loop_policy.py` → loop_iteration_runner.py/loop_history.py/loop_hitl_orchestrator.py/loop_mapper.py
- ✅ `sqlite.py` → sqlite_core.py/sqlite_ops.py/sqlite_trace.py

#### Remaining Targets

- **steps**:
    - [ ] Audit `performance_monitor.py` (695+ LOC) for split opportunities into metrics/alerts/analysis modules.
    - [ ] Audit `default_components.py` (792+ LOC) for logical grouping into serialization/caching/telemetry components.
    - [ ] Ensure all files in `application/core/` remain under 1200 LOC gate.
  
  Run: `pytest tests/architecture/test_type_safety_compliance.py::TestArchitectureCompliance::test_no_monolith_files`

```md
Goal: Maintain modular architecture with all files under 1200 LOC gate.
```

---

### 1.2 Performance-First Hot Path Optimization

- **due**: 2025-12-20
- **tags**: [performance, optimization]
- **priority**: high
- **workload**: Medium
- **guide_alignment**: ⭐⭐⭐ Perfect

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Section 11, Lesson 7):
> "Be Mindful of the Hot Path: The policy-driven architecture introduces layers of abstraction for correctness and maintainability. While generally fast, be mindful of performance in code that runs thousands of times per second."

The guide provides extensive patterns for:
- Caching results with `@lru_cache`
- Avoiding object creation in loops
- Profiling before optimizing
- Performance monitoring in production

#### Implementation Steps

- **steps**:
    - [ ] Profile `ExecutorCore.execute()` hot path using cProfile to identify bottlenecks.
    - [ ] Implement caching for repeated policy lookups in `PolicyRegistry.get()`.
    - [ ] Add object pooling for frequently created `StepResult` instances in loop execution.
    - [ ] Ensure performance thresholds are maintained (never adjusted to hide regressions per Lesson 6).
  
  Run: `pytest tests/robustness/test_performance_regression.py`

```md
Goal: Maintain <150ms budget for high-concurrency handling per existing robustness tests.
```

---

### 1.3 Testing Infrastructure Typed Fixtures

- **due**: 2025-12-18
- **tags**: [testing, typing]
- **priority**: medium
- **workload**: Small
- **guide_alignment**: ⭐⭐⭐ Strong

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Section 12.1):
> "All test fixtures must use typed factories from `tests/test_types/fixtures.py`"
> "All mock objects must use typed factories from `tests/test_types/mocks.py`"
> "Never create ad-hoc test objects with `Any` types"

#### Implementation Steps

- **steps**:
    - [ ] Audit existing tests for ad-hoc mock creation patterns.
    - [ ] Migrate tests to use `create_test_step()`, `create_test_step_result()`, `create_test_pipeline()` from fixtures.
    - [ ] Ensure all new tests use typed factories from `tests/test_types/`.
  
  Run: `make typecheck && make test-fast`

```md
Goal: 100% typed test fixtures for improved test reliability and type safety.
```

---

### 1.4 Unified Serialization with JSONObject

- **due**: 2025-12-22
- **tags**: [typing, serialization]
- **priority**: medium
- **workload**: Medium
- **guide_alignment**: ⭐⭐⭐ Strong

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Section 12.1):
> "All JSON structures must use `JSONObject` or `TypedDict` from `flujo.type_definitions`"
> "Never use `Dict[str, Any]` directly - use `JSONObject` alias"

The guide mandates this pattern throughout, with extensive examples in Sections 3, 5, and 13.

#### Implementation Steps

- **steps**:
    - [ ] Audit codebase for `Dict[str, Any]` usage patterns.
    - [ ] Replace with `JSONObject` from `flujo.type_definitions.common`.
    - [ ] Update serialization utilities to use consistent `JSONObject` typing.
    - [ ] Ensure `safe_serialize`/`safe_deserialize` maintain type consistency.
  
  Run: `make typecheck`

```md
Goal: Eliminate Dict[str, Any] usage in favor of typed JSONObject throughout.
```

---

## Phase 2: Guide-Compatible Improvements (Medium Priority)

### 2.1 Constructor Injection for Policies (Guide-Aligned DI)

- **due**: 2025-12-30
- **tags**: [architecture, di, policies]
- **priority**: medium
- **workload**: Medium
- **guide_alignment**: ⭐⭐ Partial (uses guide's existing patterns)

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Section 5, Step 3):
```python
class ExecutorCore:
    def __init__(
        self,
        agent_runner: Any,
        your_custom_step_executor: Optional[DefaultYourCustomStepExecutor] = None,
    ) -> None:
        # Type-safe policy injection with fallback to default
        self.your_custom_step_executor: DefaultYourCustomStepExecutor = (
            your_custom_step_executor or DefaultYourCustomStepExecutor()
        )
```

The guide already endorses **constructor injection for policies**. This improvement extends that pattern without introducing a full DI container (which would conflict with guide patterns).

#### What This Is NOT

❌ **Not a full DI container** - The guide uses global `get_settings()` and `ConfigManager` access
❌ **Not configuration injection** - Guide mandates global ConfigManager access

#### What This IS

✅ **Extended constructor injection** - Following the guide's existing pattern
✅ **Policy-level injection** - Making policies more testable
✅ **Factory-based composition** - Using existing `ExecutorFactory` patterns

#### Implementation Steps

- **steps**:
    - [ ] Extend `ExecutorFactory` to accept optional policy overrides.
    - [ ] Allow individual policy injection in `ExecutorCore` constructor (already partially implemented).
    - [ ] Add typed interfaces for policy dependencies (e.g., `IAgentRunner`, `ITelemetry`).
    - [ ] Maintain backward compatibility with default instantiation.
  
  Run: `pytest tests/application/core/test_policy_registry.py tests/application/core/test_executor_core.py`

```md
Goal: Extend guide-endorsed constructor injection for better testability without violating global access patterns.
```

---

### 2.2 Plugin Architecture Cleanup

- **due**: 2026-01-10
- **tags**: [architecture, plugins, skills]
- **priority**: medium
- **workload**: Medium
- **guide_alignment**: ⭐⭐ Neutral (compatible, not explicitly addressed)

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Killer Demo Skills section):
> "Web search: fetch fresh, external information without API keys."
> "Text extraction: convert unstructured text into structured JSON using an LLM."
> "Graceful degradation: if the optional dependency is missing or a search fails, returns an empty list rather than crashing."

The guide mentions skills/builtins but doesn't provide detailed plugin architecture guidance. This improvement is **compatible** with guide patterns.

#### Implementation Steps

- **steps**:
    - [ ] Document plugin/skill lifecycle in a dedicated guide section.
    - [ ] Standardize skill registration patterns across `builtins_*.py` modules.
    - [ ] Add typed interfaces for skill factories.
    - [ ] Ensure graceful degradation patterns are consistent.
  
  Run: `pytest tests/unit/test_builtins.py`

```md
Goal: Standardize plugin/skill patterns for ecosystem growth.
```

---

## Phase 3: Carefully Scoped Improvements (Lower Priority)

### 3.1 Configuration Caching and Validation (Guide-Compliant)

- **due**: 2026-01-20
- **tags**: [configuration, performance]
- **priority**: low
- **workload**: Small
- **guide_alignment**: ⭐⭐ Partial (improves without changing access patterns)

#### Justification

**FLUJO_TEAM_GUIDE.md Reference** (Section 4):
> "Canonical Source: The `flujo.infra.config_manager.ConfigManager` is the single source of truth for all configuration."
> "Accessing Settings: Use the global `get_settings()` function from `flujo.infra.settings`"

**Important**: The guide mandates **global access** through `ConfigManager` and `get_settings()`. This improvement **does not change** that pattern—it only adds caching and validation within the existing global access mechanism.

#### What This Does NOT Do

❌ **Does not inject configuration** - Guide mandates global access
❌ **Does not replace ConfigManager** - Guide says it's the "single source of truth"
❌ **Does not add a ConfigurationService** - Would violate guide patterns

#### What This Does

✅ **Adds caching within ConfigManager** - Performance improvement
✅ **Adds validation within get_settings()** - Fail-fast on invalid config
✅ **Maintains global access pattern** - Guide-compliant

#### Implementation Steps

- **steps**:
    - [ ] Add caching layer within `ConfigManager.load_config()` to avoid repeated file reads.
    - [ ] Add validation within `get_settings()` for early error detection.
    - [ ] Ensure cache invalidation on config file changes (if applicable).
  
  Run: `make test-fast`

```md
Goal: Improve configuration performance without changing access patterns.
```

---

## Explicitly Excluded (Guide Conflicts)

### ❌ Full Dependency Injection Container

**Reason**: The guide uses global access patterns (`get_settings()`, `ConfigManager`) and constructor injection for specific components. A full DI container would conflict with these established patterns.

**Guide Reference**:
> "Always access configuration through ConfigManager; do not read flujo.toml directly."
> "Use the global get_settings() function from flujo.infra.settings"

**Status**: ✅ Documented as anti-pattern in FLUJO_TEAM_GUIDE.md Section 4.

### ❌ Configuration Service Injection

**Reason**: The guide explicitly mandates global configuration access. Injecting configuration would violate this pattern.

**Alternative**: Use guide-compliant caching within existing global access (Phase 3.1).

**Status**: ✅ Documented as anti-pattern in FLUJO_TEAM_GUIDE.md Section 4.

---

## Completed: Guide Updates

### ✅ Document DI Anti-Patterns in FLUJO_TEAM_GUIDE.md

- **completed**: 2025-11-29
- **tags**: [documentation, architecture, anti-patterns]
- **priority**: high
- **workload**: Small

#### What Was Added

Added two new anti-pattern sections to FLUJO_TEAM_GUIDE.md Section 4 (Agent and Configuration Management):

1. **❌ Anti-Pattern: Full Dependency Injection Container**
   - Explains why a global DI container is forbidden
   - Documents the correct alternative: constructor injection for policies
   - Includes code examples showing wrong vs. correct patterns

2. **❌ Anti-Pattern: Configuration Service Injection**
   - Explains why configuration injection is forbidden
   - Documents the correct alternative: global `get_settings()` access
   - Includes code examples showing wrong vs. correct patterns

#### Justification

These anti-patterns were identified during architectural analysis and needed explicit documentation to prevent future violations. The guide already documents many anti-patterns (Sections 1, 2, 10, 13), so adding these follows established precedent.

```md
Goal: Prevent architectural drift by explicitly documenting forbidden patterns.
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Monolith files (>1200 LOC) | 0 | 0 | `test_no_monolith_files` |
| Type errors | 0 | 0 | `make typecheck` |
| Performance budget | <150ms | <150ms | `test_high_concurrency_handling` |
| Test pass rate | 100% | 100% | `make test-fast` |
| Dict[str, Any] occurrences | TBD | 0 | grep audit |

---

## Implementation Order

1. **Immediate** (Phase 1): Modularization, Performance, Testing Fixtures, JSONObject
2. **Medium-term** (Phase 2): Constructor Injection, Plugin Cleanup
3. **Long-term** (Phase 3): Configuration Caching

---

## Validation Checklist

Before implementing any change, verify:

- [ ] Does `make all` pass with 0 errors?
- [ ] Does the change follow FLUJO_TEAM_GUIDE.md patterns?
- [ ] Are all function signatures fully typed?
- [ ] Did I use `JSONObject` instead of `Dict[str, Any]`?
- [ ] Did I follow the policy-driven architecture?
- [ ] Did I handle control flow exceptions correctly?

---

## References

- `FLUJO_TEAM_GUIDE.md` - Primary architectural guide
- `Kanban/phase3.md` - Policy decoupling status (completed)
- `Kanban/fix_items.md` - Gate blockers status (completed)
- `docs/development/type_safety.md` - Type safety patterns

