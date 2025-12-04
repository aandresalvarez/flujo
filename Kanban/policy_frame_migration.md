# Product Requirements Document: Policy Frame-Only Migration
**Strategy:** Vertical Slice Refactoring with Runtime Adaptation

## 1. Executive Summary
This initiative refactors the Flujo orchestration engine to replace the "Function Argument Soup" pattern with a **Parameter Object Pattern**. We will migrate all `StepPolicy.execute` signatures from accepting loose arguments (step, data, context, resources, etc.) to accepting a single, unified `ExecutionFrame[Any]` object.

**Why:**
1.  **Extensibility:** Adding execution parameters (e.g., `trace_id`, `priority`) no longer requires changing signatures across the entire class hierarchy.
2.  **State Centralization:** The `ExecutionFrame` becomes the canonical snapshot of execution, simplifying features like Pause/Resume and Replay.
3.  **Type Safety:** Reduces `Any` usage and enforces strict contracts between the Core and Policies.

---

## 2. Architecture: The Frame Pattern

### Current State (Anti-Pattern)
Policies depend on specific method signatures. Adding a parameter requires touching every policy.
```python
# Fragile
async def execute(self, core, step, data, context, resources, limits, stream, ...)
```

### Target State (Robust)
Policies depend on a Context Object. The Core passes the frame; the Policy extracts what it needs.
```python
# Robust
async def execute(self, core: ExecutorCore, frame: ExecutionFrame[Any]) -> StepOutcome
```

---

## 3. Migration Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Breaking External Plugins** | High | **Runtime Adapter:** The Dispatcher will inspect custom policies at runtime. If they use the old signature, it will unpack the frame automatically. |
| **"Shotgun Surgery"** | High | **Vertical Slicing:** We will migrate one policy type at a time (e.g., `SimpleStep` -> verify -> `ParallelStep` -> verify), ensuring the build stays green. |
| **Recursion State Corruption** | Critical | **Frame Factories:** Container steps (`Loop`, `Parallel`) must create *new* frames for children, never mutate and pass the parent frame. |
| **Test Fragility** | Medium | **Attribute Assertions:** Tests will be updated to assert on specific frame attributes, not Frame object equality (which is flaky). |

---

## 4. Implementation Plan

### Phase 0: Foundation & Compatibility Adapter
**Goal:** Enable the engine to support *both* new and old signatures simultaneously.

1.  **Harden `ExecutionFrame`:** Ensure `make_execution_frame` correctly handles all edge cases (quota propagation, stream flags).
2.  **Update `StepPolicy` Protocol:** Modify the Protocol definition to allow `ExecutionFrame` without breaking existing implementations immediately (use `Union` or `*args` temporarily).
3.  **Implement Dispatcher Adapter:** Modify `ExecutionDispatcher.dispatch` to inspect the target policy before calling it.
    *   *Logic:*
        ```python
        # Pseudocode for Adapter Logic
        import inspect
        sig = inspect.signature(policy.execute)
        if "frame" in sig.parameters:
            # New Style
            return await policy.execute(core, frame)
        else:
            # Legacy Style (Backwards Compatibility)
            return await policy.execute(
                core,
                frame.step,
                frame.data,
                frame.context,
                # ... unpack remaining frame attributes
            )
        ```

### Phase 1: Vertical Migration (Iterative)
**Goal:** Migrate internal policies one by one.

**Order of Operations:**
1.  `SimpleStepExecutor` (The baseline)
2.  `AgentStepExecutor` (The most common)
3.  `CacheStepExecutor`
4.  `ConditionalStepExecutor`
5.  `ImportStepExecutor`

**Tasks for each Policy:**
*   **Refactor Policy:** Change `execute` signature to `(core, frame)`. Remove local arg unpacking.
*   **Refactor Handler:** Update `StepHandler` to construct a `ExecutionFrame` before calling the policy.
*   **Update Unit Tests:** Update tests specific to this policy.
*   **Validation:** Run `make test-fast` to ensure no regression.

### Phase 2: Complex Step Migration (Recursion Safety)
**Goal:** Migrate container steps that spawn child executions.

**Policies:** `ParallelStepExecutor`, `LoopStepExecutor`, `DynamicRouterStepExecutor`, `HitlStepExecutor`.

**Critical Requirement:**
When a container step executes children, it must instantiate a **new** `ExecutionFrame` for the child using `core._make_execution_frame`.
*   *Anti-Pattern:* `child_policy.execute(core, self.frame)` (Context pollution).
*   *Correct:* `child_frame = core._make_execution_frame(...); child_policy.execute(core, child_frame)`.

### Phase 3: Test Suite Stabilization
**Goal:** Fix test fragility caused by the signature change.

1.  **Audit Mocks:** Find tests mocking `ExecutorCore.execute`.
2.  **Refactor Assertions:** Stop asserting `called_with(specific_frame_instance)`.
    *   *Change to:* Capture the `call_args`, extract the frame object, and assert `frame.data == expected` and `frame.step.name == expected`.
3.  **Verify Dynamic Logic:** Ensure dynamic router and HITL tests still correctly assert pause/resume semantics via the Dispatcher.

### Phase 4: Cleanup & Finalization
**Goal:** Remove the scaffolding.

1.  **Remove Legacy Shims:** Delete `_handle_*`, `_execute_*`, and `execute_step_compat` from `ExecutorCore`.
2.  **Strict Protocol:** Update `StepPolicy` Protocol to *only* accept `ExecutionFrame`.
3.  **Remove Adapter (Internal):** Remove the inspection logic from `ExecutionDispatcher` for internal paths.
    *   *Note:* You may choose to keep the adapter logic active *only* for external custom policies if you wish to support v0.4 plugins in v0.5.

---

## 5. Validation & Exit Criteria

1.  **Signature Consistency:** All internal policies expose `execute(core, frame: ExecutionFrame)`.
2.  **No Legacy Branches:** `ExecutorCore` contains no `if isinstance(arg, ExecutionFrame)` checks (except in the Dispatcher adapter).
3.  **Green Suite:** `make test` passes.
4.  **Type Safety:** `make typecheck` passes with strict typing on the `ExecutionFrame`.

## 6. Rollback Plan

If `make test` fails significantly during Phase 1 or 2:
1.  Revert the specific Policy change.
2.  The **Dispatcher Adapter** (Phase 0) ensures the rest of the system continues to function even if one policy remains legacy. This prevents a "broken main" situation.