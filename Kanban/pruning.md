# Product Requirements Document (PRD): Progressive Optimization Layer Pruning

| Project | Flujo |
| :--- | :--- |
| **Version** | 3.2 (Architecturally Verified) |
| **Status** | **Ready for Implementation** |
| **Feature** | Core Cleanup & Dead Code Removal |
| **Strategy** | **Strangler Fig Pattern with Subprocess Verification** |

---

## 1. Executive Summary

This initiative aims to remove the unused "optimization" layer within the Flujo core architecture (`adaptive_resource_manager`, `circuit_breaker`, `performance_monitor`, etc.). This code is currently inactive in default execution paths but adds significant maintenance overhead, threading complexity, and a heavy binary dependency (`psutil`).

We will execute a **Progressive Decoupling Strategy**. We will first refactor consuming components to sever dependencies, prove via **isolated subprocess tests** that the modules are not loaded, and only then proceed to delete the code. Crucially, we will retain a compatibility stub to prevent import errors for legacy integrations.

---

## 2. Problem Statement

1.  **Dead Weight**: The `flujo/application/core/optimization/` directory contains complex logic that is not wired into the default `SimpleStepExecutor`.
2.  **Heavy Dependencies**: The unused modules import `psutil`, which requires compilation and causes issues in serverless/Lambda environments.
3.  **Implicit Coupling**: `BackgroundTaskManager` currently imports `OptimizedErrorHandler`, forcing the entire optimization tree to load.
4.  **API Confusion**: `ExecutorCore` accepts an `optimization_config` argument that implies functionality which is effectively dormant.

---

## 3. Objectives

1.  **Eliminate `psutil` dependency**: Remove the requirement for system-level resource monitoring.
2.  **Zero API Breakage**: Maintain backward compatibility for `ExecutorCore` initialization.
3.  **Verified Isolation**: Implement **subprocess-based** tests proving that optimization modules are not loaded.
4.  **Safe Code Deletion**: Remove the `optimization/` directory while preserving `optimization_config_stub.py`.

---

## 4. Phase 1: Verification Framework

To accurately verify that modules are not loaded, we must run tests in a fresh process. Standard `sys.modules` manipulation within a single test runner is insufficient due to singleton module caching.

### 4.1 Task: Create Subprocess Isolation Test
**File:** `tests/architecture/test_module_isolation.py`

Implement a test that spawns a clean Python process, runs a minimal pipeline, and inspects `sys.modules`.

```python
import sys
import subprocess
from pathlib import Path

def test_standard_run_does_not_import_psutil_or_optimization():
    """
    Verifies that a fresh Flujo process does NOT load the optimization layer
    or the heavy psutil dependency.
    """
    # Python script to run in the subprocess
    code = """
import sys
from flujo import Pipeline, Step, Flujo

# 1. Define and run a minimal pipeline
step = Step.from_callable(lambda x: x, name="noop")
pipeline = Pipeline.from_step(step)
runner = Flujo(pipeline)
runner.run(42)

# 2. Inspect loaded modules
forbidden = ['psutil', 'flujo.application.core.optimization']
loaded = [m for m in sys.modules.keys() if any(f in m for f in forbidden)]

if loaded:
    print(f"VIOLATION: Found forbidden modules: {loaded}")
    sys.exit(1)

print("SUCCESS")
"""
    
    # Run the subprocess
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True
    )

    # Assert success
    if result.returncode != 0:
        pytest.fail(f"Isolation test failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
```

**Expectation:** This test **MUST FAIL** initially.

---

## 5. Phase 2: Surgical Decoupling

Sever the connections between Core and Optimization.

### 5.1 Refactor `BackgroundTaskManager`
**Goal**: Remove dependency on `OptimizedErrorHandler` while improving error classification robustness.

**File:** `flujo/application/core/background_task_manager.py`

1.  **Remove Import**: Delete `from .optimized_error_handler import ...`.
2.  **Implement Robust Error Classification**: Instead of relying on fragile string matching alone, prioritize type checks for standard library exceptions, falling back to string matching only for un-importable types.

    *Replace lines ~284-293 with:*
    ```python
    # OLD: ErrorClassifier().classify_error(...)
    
    # NEW:
    import asyncio
    
    try:
        # 1. Priority: Type-based classification (Robust)
        if isinstance(e, (asyncio.TimeoutError, ConnectionError)):
            metadata["error_category"] = "network"
        elif isinstance(e, (ValueError, TypeError, AttributeError)):
             metadata["error_category"] = "validation"
        elif isinstance(e, (PermissionError, OSError)):
             metadata["error_category"] = "system"
        
        # 2. Fallback: String-based classification (Legacy/Third-party)
        else:
            err_str = str(e).lower()
            err_name = type(e).__name__.lower()
            
            if "auth" in err_str or "auth" in err_name:
                metadata["error_category"] = "authentication"
            elif "quota" in err_str or "limit" in err_str:
                metadata["error_category"] = "resource_exhaustion"
            else:
                metadata["error_category"] = "unknown"
                
        if final_context is not None and hasattr(final_context, "scratchpad"):
            final_context.scratchpad["background_error_category"] = metadata["error_category"]
    except Exception:
        metadata["error_category"] = "unknown"
    ```

3.  **Update Constants**: Replace `ErrorCategory.CONTROL_FLOW.value` with literal `"control_flow"`.

### 5.2 Refactor `ExecutorCore`
**Goal**: Remove imports but keep the `__init__` signature valid using the Stub.

**File:** `flujo/application/core/executor_core.py`

1.  **Remove Imports**: Delete imports referencing `flujo.application.core.optimization`.
2.  **Keep Stub Import**: Ensure `from .optimization_config_stub import OptimizationConfig` remains or is added if missing.
3.  **Update `__init__`**:
    ```python
    def __init__(
        self, 
        ..., 
        # Type hint remains valid via the stub
        optimization_config: Optional[OptimizationConfig] = None, 
        ...
    ):
        # ... existing code ...
        
        # Remove logic that uses optimization_config
        if optimization_config is not None:
            import warnings
            warnings.warn(
                "optimization_config is deprecated and has no effect.", 
                DeprecationWarning, stacklevel=2
            )
        
        # Hardcode the flag that previously relied on config
        self.enable_optimized_error_handling = False 
    ```

### 5.3 Refactor `ExecutorFactory`
**File:** `flujo/application/core/factories.py`

1.  Clean up `create_executor` to stop passing the config object down.

### 5.4 Verify Isolation
**Action**: Run `tests/architecture/test_module_isolation.py`.
**Expectation**: The test **MUST PASS**. `psutil` should no longer be loaded.

---

## 6. Phase 3: Safe Deletion

### 6.1 File Deletion
**Critical Rule:** Delete the *folder*, but **preserve the stub**.

1.  **Retain**: `flujo/application/core/optimization_config_stub.py` (Do NOT delete).
2.  **Delete Directory**: `flujo/application/core/optimization/` and all contents:
    *   `adaptive_resource_manager.py`
    *   `circuit_breaker.py`
    *   `performance_monitor.py`
    *   `optimized_error_handler.py`
    *   (and others in that folder)

### 6.2 Dependency Cleanup
**File:** `pyproject.toml`
*   Remove `psutil` from dependencies.

### 6.3 Test Suite Cleanup
*   Delete tests in `tests/unit/application/core/optimization/`.
*   Scan `tests/` for `from flujo.application.core.optimization import ...` and remove/refactor those tests.

---

## 7. Phase 4: Artifact Retention & Deprecation

To prevent breaking users who explicitly import the config class:

1.  **Stub Verification**: Ensure `flujo/application/core/optimization_config_stub.py` contains a class `OptimizationConfig` that accepts `**kwargs`, does nothing, and issues a warning on init.
2.  **Documentation**: Update `CHANGELOG.md`:
    > **Deprecated:** The `optimization` layer has been removed to reduce bloat. `ExecutorCore` still accepts `optimization_config` for backward compatibility, but it is ignored. The `psutil` dependency has been dropped.

---

## 8. Rollback Strategy

1.  **Subprocess Test Failure**: If Phase 2 verification fails, grep for `flujo.application.core.optimization`. A "zombie import" likely remains in `__init__.py` or type hints.
2.  **AttributeError**: If runtime fails with `AttributeError: 'ExecutorCore' object has no attribute 'optimization_config'`, ensure the attribute was not being accessed by legacy properties in `ExecutorCore`.
3.  **Stub Import Error**: If users report `ImportError`, restore `optimization_config_stub.py` immediately.

---

## 9. Success Metrics

*   **Verification**: `tests/architecture/test_module_isolation.py` passes.
*   **Weight**: `psutil` is gone from the lockfile.
*   **Compatibility**: Existing user code initializing `Flujo(optimization_config=...)` still runs (albeit with a warning).