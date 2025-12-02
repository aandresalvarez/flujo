"""Module isolation tests.

These tests verify that standard pipeline execution does not import
unused optimization modules, ensuring clean separation of concerns
and avoiding unnecessary dependencies.
"""

import sys
import builtins
from typing import Set
import pytest

# Mark as slow since it involves import tracking
pytestmark = [pytest.mark.slow, pytest.mark.timeout(60)]


def test_standard_run_does_not_import_optimization() -> None:
    """
    Verifies that initializing and running a standard pipeline
    does NOT trigger imports of the optimization layer or psutil.

    This test tracks imports during execution to catch lazy imports
    that might not appear in sys.modules immediately.

    **Expected Behavior**: This test MUST FAIL initially, proving that
    optimization modules are currently being imported. After Phase 2
    decoupling, it should PASS.
    """
    # 1. Clean slate: Force unload modules if they exist
    modules_to_remove: list[str] = []
    for mod in list(sys.modules.keys()):
        if "flujo.application.core.optimization" in mod or mod == "psutil":
            modules_to_remove.append(mod)
    for mod in modules_to_remove:
        del sys.modules[mod]

    # 2. Track imports during execution
    original_import = builtins.__import__
    imported_modules: Set[str] = set()

    def tracking_import(name: str, *args: object, **kwargs: object) -> object:
        if "flujo.application.core.optimization" in name or name == "psutil":
            imported_modules.add(name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = tracking_import  # type: ignore[assignment]

    try:
        # 3. Perform Standard Run
        from flujo import Pipeline, Step, Flujo

        async def simple_step(x: int) -> int:
            return x

        step = Step.from_callable(simple_step, name="noop")
        pipeline = Pipeline.from_step(step)
        runner = Flujo(pipeline)
        result = runner.run(42)  # Actually execute the pipeline

        # 4. Verification
        loaded_opt = [m for m in sys.modules if "flujo.application.core.optimization" in m]
        # Check if psutil was imported DURING execution (not just if it exists from test deps)
        psutil_imported_during_execution = any("psutil" in name for name in imported_modules)

        assert not loaded_opt, f"Optimization modules incorrectly loaded: {loaded_opt}"
        assert not psutil_imported_during_execution, (
            f"psutil imported during execution: {[m for m in imported_modules if 'psutil' in m]}"
        )
        assert not imported_modules, (
            f"Optimization modules imported during execution: {imported_modules}"
        )

        # Verify the pipeline actually ran successfully
        assert result is not None
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]
