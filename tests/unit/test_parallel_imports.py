"""Tests for flujo.application.parallel module imports."""

from flujo.application.parallel import ExecutorCore, StepExecutor


def test_parallel_imports():
    """Test that parallel module imports work correctly."""
    # This test ensures the imports are working and the module is accessible
    assert ExecutorCore is not None
    assert StepExecutor is not None

    # Verify these are callable/importable
    assert hasattr(ExecutorCore, "__call__") or hasattr(ExecutorCore, "__class__")
    assert hasattr(StepExecutor, "__call__") or hasattr(StepExecutor, "__class__")
