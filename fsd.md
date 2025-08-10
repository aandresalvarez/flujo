

### **FSD 11 of 11: Final Deprecation and Cleanup**

**1. Rationale & First Principles**

*   **Goal:** To complete the refactor by removing all obsolete code, thereby simplifying the codebase, eliminating potential for confusion, and solidifying the new architecture as the single source of truth.
*   **Why:** This upholds the **Maintainability** principle. Dead code is a significant source of technical debt. It adds cognitive load for developers, can be mistakenly used or "fixed" during future work, and clutters the project, making it harder to navigate and understand. This final step ensures the codebase is clean, lean, and reflects only the new, correct architecture.

**2. Scope of Work**

This FSD is a "search and destroy" mission for all code that has been made redundant by the new `ExecutorCore`.

1.  **Delete `flujo/application/core/step_logic.py`:**
    *   The file is now completely unused. The `ExecutorCore` contains all of its logic. It will be deleted from the repository.

2.  **Remove Backward Compatibility Wrappers:**
    *   In `flujo/application/core/ultra_executor.py`, the following backward-compatibility classes at the bottom of the file will be deleted:
        *   `UltraStepExecutor`
        *   `_Frame`
        *   `_LRUCache`
        *   `_UsageTracker`

3.  **Clean Up `flujo/application/runner.py`:**
    *   The `_run_step` method is now a complex wrapper around the new `backend.execute_step` call. It can be significantly simplified or removed. The `_execute_steps` method in `ExecutionManager` is the primary loop.
    *   Search for any remaining imports from `step_logic.py` and remove them.

4.  **Clean Up `flujo/application/core/execution_manager.py`:**
    *   The `step_executor` parameter in `execute_steps` is a legacy holdover. It can now be removed, as the manager will always use `self.backend`.

**3. Implementation Details**

#### **Task 1: Delete `step_logic.py`**

*   **Action:** Delete the file `flujo/application/core/step_logic.py`.
*   **Verification:** After deletion, run a project-wide search for `step_logic`. The only remaining references should be in `git` history, not in the active codebase.

#### **Task 2: Remove Backward Compatibility Wrappers**

*   **Action:** In `flujo/application/core/ultra_executor.py`, delete the class definitions for `UltraStepExecutor`, `_Frame`, `_LRUCache`, and `_UsageTracker`.
*   **Verification:** Check for any import errors. Tests that specifically targeted these classes will need to be deleted or refactored to test the new components they were wrapping (e.g., `InMemoryLRUBackend`).

#### **Task 3 & 4: Clean Up `runner.py` and `execution_manager.py`**

The `Flujo` runner's `_execute_steps` method currently passes `self._run_step` as the `step_executor` to the `ExecutionManager`. This is an unnecessary layer of indirection. We can simplify this flow.

```python
# In flujo/application/runner.py

class Flujo(Generic[RunnerInT, RunnerOutT, ContextT]):
    # ...

    # ⛔️ DELETE this entire method. It is a legacy wrapper.
    # async def _run_step(...):
    #     ...

    async def _execute_steps(
        self,
        start_idx: int,
        data: Any,
        context: Optional[ContextT],
        result: PipelineResult[ContextT],
        # ... other params
    ) -> AsyncIterator[Any]:
        """Execute pipeline steps using the new execution manager."""
        assert self.pipeline is not None

        state_manager: StateManager[ContextT] = StateManager[ContextT](self.state_backend)
        usage_governor: UsageGovernor[ContextT] = UsageGovernor[ContextT](self.usage_limits)
        step_coordinator: StepCoordinator[ContextT] = StepCoordinator[ContextT](
            self.hooks, self.resources
        )

        execution_manager = ExecutionManager(
            self.pipeline,
            backend=self.backend, # ✅ Pass the backend directly.
            state_manager=state_manager,
            usage_governor=usage_governor,
            step_coordinator=step_coordinator,
        )

        # ✅ The `step_executor` parameter is no longer needed and is removed.
        async for item in execution_manager.execute_steps(
            start_idx=start_idx,
            data=data,
            context=context,
            result=result,
            stream_last=stream_last,
            run_id=run_id,
            state_created_at=state_created_at,
        ):
            yield item
```

```python
# In flujo/application/core/execution_manager.py

class ExecutionManager(Generic[ContextT]):
    def __init__(
        self,
        pipeline: Pipeline[Any, Any],
        *,
        backend: Optional[ExecutionBackend] = None, # ✅ This is now the source of execution
        # ... other components
    ):
        # ...
        self.backend = backend
        # ...

    async def execute_steps(
        self,
        # ... other params
        # ⛔️ DELETE the `step_executor` parameter.
    ) -> AsyncIterator[Any]:
        # ...
        # Inside the loop over steps:
        # ⛔️ OLD: async for item in self.step_coordinator.execute_step(..., step_executor=step_executor):
        # ✅ NEW: The coordinator now gets the backend directly.
        async for item in self.step_coordinator.execute_step(
            step=step,
            data=data,
            context=context,
            backend=self.backend, # Pass the backend to the coordinator
            # ... other params
        ):
            # ...
```

**4. Testing Strategy**

*   **Static Analysis:** After deleting the files and code, run linters (`ruff`, `flake8`) and type checkers (`mypy`). This is the most important step to catch any broken imports or dangling references to the deleted code.
*   **Regression Tests (Existing):**
    *   This is the final, ultimate validation. Run the **entire test suite** one last time.
    *   A 100% pass rate is the definitive sign-off that the refactor is complete, correct, and has not introduced any regressions. The system is now fully running on the new, clean architecture.

**5. Acceptance Criteria**

*   [ ] The file `flujo/application/core/step_logic.py` has been deleted.
*   [ ] The backward-compatibility wrapper classes in `ultra_executor.py` have been deleted.
*   [ ] The `_run_step` method in `runner.py` has been removed, and `_execute_steps` is simplified.
*   [ ] The `step_executor` parameter has been removed from `ExecutionManager.execute_steps`.
*   [ ] The project passes all static analysis checks (`mypy`, linters) with no errors related to the removed code.
*   [ ] **100% of the entire test suite passes**, finalizing the refactor.
