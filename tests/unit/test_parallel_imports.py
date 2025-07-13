"""Tests for flujo.application.parallel module imports."""

from flujo.application.parallel import _execute_parallel_step_logic, StepExecutor


def test_parallel_imports():
    """Test that parallel module imports work correctly."""
    # This test ensures the imports are working and the module is accessible
    assert _execute_parallel_step_logic is not None
    assert StepExecutor is not None

    # Verify these are callable/importable
    assert callable(_execute_parallel_step_logic)
    assert hasattr(StepExecutor, "__call__") or hasattr(StepExecutor, "__class__")
