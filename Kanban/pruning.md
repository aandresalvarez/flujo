# Product Requirements Document (PRD): Progressive Optimization Layer Pruning

| Project | Flujo |
| :--- | :--- |
| **Version** | 3.1 (Enhanced) |
| **Status** | **Approved** |
| **Feature** | Core cleanup & Dead Code Removal |
| **Strategy** | **Test-Driven Decoupling** |

---

## 1. Executive Summary

This initiative aims to remove the unused "optimization" layer within the Flujo core architecture (`adaptive_resource_manager`, `circuit_breaker`, `performance_monitor`, etc.). This code is currently inactive in default execution paths but adds significant maintenance overhead, threading complexity, and a heavy binary dependency (`psutil`).

We will execute a **Progressive Decoupling Strategy**. Instead of immediately deleting files, we will first refactor the consuming components (`ExecutorCore`, `BackgroundTaskManager`) to sever their dependencies. We will prove via architectural tests that the optimization modules are no longer loaded at runtime, and only then proceed to delete the code. This guarantees zero API breakage and system stability during the transition.

---

## 2. Problem Statement

1.  **Dead Weight**: The `flujo/application/core/optimization/` directory contains complex logic for circuit breaking and resource management that is not wired into the default `SimpleStepExecutor`.
2.  **Heavy Dependencies**: The unused modules import `psutil`, which requires compilation and causes issues in serverless/Lambda environments.
3.  **Implicit Coupling**: `BackgroundTaskManager` currently imports `OptimizedErrorHandler` solely for error classification enums, forcing the entire optimization tree to load even when unused.
4.  **API Confusion**: `ExecutorCore` accepts an `optimization_config` argument that implies functionality which is effectively dormant.

---

## 3. Objectives

1.  **Eliminate `psutil` dependency**: Remove the requirement for system-level resource monitoring libraries.
2.  **Zero API Breakage**: Maintain backward compatibility for `ExecutorCore` initialization signature.
3.  **Verified Isolation**: Implement tests proving that standard pipeline runs do not import optimization modules.
4.  **Safe Code Deletion**: Remove ~12 files/modules only after decoupling is verified.

---

## 4. Phase 1: Verification Framework

Before modifying application code, we establish a test to detect if the optimization layer is being loaded.

### 4.1 Task: Create Isolation Probe Test
**File:** `tests/architecture/test_module_isolation.py`

Implement a test that performs a standard pipeline run and asserts that specific modules are **not** present in `sys.modules`. The test uses import tracking to catch lazy imports that might not appear in `sys.modules` immediately.

```python
def test_standard_run_does_not_import_optimization():
    """
    Verifies that initializing and running a standard pipeline 
    does NOT trigger imports of the optimization layer or psutil.
    
    This test tracks imports during execution to catch lazy imports
    that might not appear in sys.modules immediately.
    """
    import sys
    import importlib
    
    # 1. Clean slate: Force unload modules if they exist
    modules_to_remove = []
    for mod in list(sys.modules.keys()):
        if "flujo.application.core.optimization" in mod or mod == "psutil":
            modules_to_remove.append(mod)
    for mod in modules_to_remove:
        del sys.modules[mod]
    
    # 2. Track imports during execution
    original_import = __builtins__.__import__
    imported_modules = set()
    
    def tracking_import(name, *args, **kwargs):
        if "flujo.application.core.optimization" in name or name == "psutil":
            imported_modules.add(name)
        return original_import(name, *args, **kwargs)
    
    __builtins__.__import__ = tracking_import
    
    try:
        # 3. Perform Standard Run
        from flujo import Pipeline, Step, Flujo
        step = Step.from_callable(lambda x: x, name="noop")
        pipeline = Pipeline.from_step(step)
        runner = Flujo(pipeline)
        result = runner.run(42)  # Actually execute the pipeline
        
        # 4. Verification
        loaded_opt = [m for m in sys.modules if "flujo.application.core.optimization" in m]
        loaded_psutil = [m for m in sys.modules if "psutil" in m]
        
        assert not loaded_opt, f"Optimization modules incorrectly loaded: {loaded_opt}"
        assert not loaded_psutil, f"psutil incorrectly loaded: {loaded_psutil}"
        assert not imported_modules, f"Optimization modules imported during execution: {imported_modules}"
    finally:
        __builtins__.__import__ = original_import
```

**Expectation:** This test **MUST FAIL** initially, proving that optimization modules are currently being imported.

---

## 5. Phase 2: Surgical Decoupling

We remove the import hooks connecting the Core to the Optimization layer.

### 5.1 Refactor `BackgroundTaskManager`
**Goal**: Remove the dependency on `OptimizedErrorHandler` enums (`ErrorCategory`) by replacing them with a local heuristic.

**File:** `flujo/application/core/background_task_manager.py`

1.  **Remove Import**: Delete `from .optimized_error_handler import ...`.
2.  **Replace Error Categorization**:
    Replace the complex `ErrorClassifier` usage with a simple string matching block in `_execute_background_step` (lines ~284-293).
3.  **Update Constants**: Replace ALL usages of `ErrorCategory.CONTROL_FLOW.value` with the string literal `"control_flow"`:
    - **Line ~254**: In `_mark_background_task_paused` handler (PausedException case)
    - **Line ~269**: In `_execute_background_step` PipelineAbortSignal handler
    
    ```python
    # OLD (both locations):
    metadata["error_category"] = ErrorCategory.CONTROL_FLOW.value
    
    # NEW (both locations):
    metadata["error_category"] = "control_flow"
    ```

4.  **Complete Error Classification Replacement**: In the general Exception handler (lines ~284-293), replace the entire error classification block:
    
    ```python
    # OLD:
    try:
        err_ctx = ErrorContext.from_exception(e)
        ErrorClassifier().classify_error(err_ctx)
        metadata["error_category"] = err_ctx.category.value
        if final_context is not None and hasattr(final_context, "scratchpad"):
            final_context.scratchpad["background_error_category"] = err_ctx.category.value
    except Exception:
        pass
    
    # NEW:
    try:
        err_name = type(e).__name__
        err_str = str(e).lower()
        if "Timeout" in err_name or "timeout" in err_str:
            metadata["error_category"] = "timeout"
        elif "Connection" in err_name or "connection" in err_str or "network" in err_str:
            metadata["error_category"] = "network"
        elif "Validation" in err_name or "validation" in err_str:
            metadata["error_category"] = "validation"
        elif "Auth" in err_name or "auth" in err_str:
            metadata["error_category"] = "authentication"
        else:
            metadata["error_category"] = "unknown"
        if final_context is not None and hasattr(final_context, "scratchpad"):
            final_context.scratchpad["background_error_category"] = metadata["error_category"]
    except Exception:
        metadata["error_category"] = "unknown"
    ```

### 5.2 Refactor `ExecutorCore`
**Goal**: Maintain API signature compatibility but stop importing optimization configs.

**File:** `flujo/application/core/executor_core.py`

1.  **Remove Imports**: Delete imports referencing `optimization.config` and `performance_monitor`.
2.  **Update `__init__` Signature**:
    Change the type hint for `optimization_config` to `Any` (or `object`) to avoid needing the class import.
    ```python
    def __init__(self, ..., optimization_config: Any = None, ...):
    ```
3.  **Update `__init__` Body**:
    Remove the validation and coercion logic for the config. Add a deprecation warning if it is used.
    ```python
    # Hardcode flag to False (standard handling)
    self.enable_optimized_error_handling = False
    
    if optimization_config is not None:
        import warnings
        warnings.warn(
            "optimization_config is deprecated and has no effect.", 
            DeprecationWarning, stacklevel=2
        )
    ```
4.  **Delete Methods**: Remove methods exposing optimization stats (e.g., `get_optimization_stats`, `export_config`).

### 5.3 Refactor `ExecutorFactory`
**File:** `flujo/application/core/factories.py`

1.  Remove `optimization_config` from `__init__` parameter list.
2.  Remove `self._optimization_config = optimization_config` from `__init__` body.
3.  Remove `optimization_config=self._optimization_config` from `create_executor` method's `ExecutorCore(...)` call.

### 5.4 Cleanup `agent_policy_run.py`
**File:** `flujo/application/core/policies/agent_policy_run.py`

**Goal**: Remove optional imports of `optimized_telemetry` that keep the module alive.

1.  **Remove Optional Imports**: Delete all `try/except` blocks that import `optimized_telemetry`:
    
    ```python
    # DELETE all instances of:
    # try:
    #     from flujo.application.core.optimized_telemetry import increment_counter as _inc
    #     _inc("estimator.usage", 1, tags={"strategy": strategy_name})
    # except Exception:
    #     pass
    ```
    
    **Note**: These are optional metrics for telemetry. Removing them has no functional impact on execution.

### 5.5 Verify Isolation
**Action**: Run the `test_module_isolation.py` created in Phase 1.
**Expectation**: The test **MUST PASS** now. This confirms safe decoupling.

---

## 6. Phase 3: Safe Deletion

With isolation verified, we can safely remove the dead code.

### 6.1 File Deletion
Remove the following files and directories:

*   **Directory**: `flujo/application/core/optimization/`
*   **Files**:
    *   `flujo/application/core/adaptive_resource_manager.py`
    *   `flujo/application/core/performance_monitor.py`
    *   `flujo/application/core/performance_models.py`
    *   `flujo/application/core/load_balancer.py`
    *   `flujo/application/core/graceful_degradation.py`
    *   `flujo/application/core/circuit_breaker.py`
    *   `flujo/application/core/error_recovery_strategies.py`
    *   `flujo/application/core/optimization_parameter_tuner.py`
    *   `flujo/application/core/optimized_error_handler.py`
    *   `flujo/application/core/optimized_telemetry.py`
    *   `flujo/application/core/telemetry_models.py`

### 6.2 Dependency Cleanup
**File:** `pyproject.toml`
*   Remove `psutil` from project dependencies.

### 6.3 Test Suite Cleanup
*   Delete unit tests residing in `tests/unit/application/core/optimization/`.
*   Delete integration tests: `tests/integration/test_executor_core_optimization_integration.py` and `tests/regression/test_executor_core_optimization_regression.py`.
*   Scan remaining integration tests for imports of `OptimizationConfig` and remove them.

---

## 7. Phase 4: Deprecation Period (Optional)

If maintaining strict backward compatibility is critical for external consumers:

### 7.1 Deprecation Strategy
1. **Keep deleted code in a separate branch** (`archive/optimization-layer`) for one release cycle.
2. **Document migration path** in `CHANGELOG.md`:
   ```markdown
   ## Deprecated
   - `ExecutorCore(optimization_config=...)` parameter is deprecated and has no effect.
     It will be removed in v0.8.0. Use standard error handling (default behavior).
   ```
3. **Monitor deprecation warnings** in production to identify external users.
4. **Remove in next major version** (e.g., v0.8.0) by removing the parameter entirely.

### 7.2 Alternative: Immediate Removal
If the codebase is internal-only or breaking changes are acceptable:
- Skip Phase 4 and proceed directly to complete removal in Phase 3.
- Update version to v0.8.0 to signal breaking changes.

---

## 8. Rollback Strategy

1.  **Decoupling Failure**: If `test_module_isolation.py` fails in Phase 2, analyze the import chain using `grep` to find the remaining link to the optimization folder. Do not proceed to deletion.
2.  **Runtime Errors**: If the new error heuristic in `BackgroundTaskManager` causes crashes, revert that specific file to its previous state. The optimization code remains present (but unused) until the fix is applied.
3.  **API Incompatibility**: If external consumers report `TypeError` on `ExecutorCore`, ensure the `optimization_config` argument was correctly preserved as `Any` in `__init__`.

---

## 9. Success Metrics

*   `tests/architecture/test_module_isolation.py` passes.
*   Full test suite (`make test`) passes.
*   `psutil` is no longer installed in the environment.
*   `flujo run` executes successfully on a basic pipeline.