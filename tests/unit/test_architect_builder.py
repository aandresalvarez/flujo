from __future__ import annotations

import os


def test_build_architect_pipeline_minimal() -> None:
    # Ensure clean environment for consistent test behavior
    original_env = {}
    env_vars_to_clear = [
        "FLUJO_ARCHITECT_STATE_MACHINE",
        "FLUJO_ARCHITECT_IGNORE_CONFIG",
        "FLUJO_TEST_MODE",
    ]

    # Clear potentially interfering environment variables
    for var in env_vars_to_clear:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    try:
        from flujo.architect.builder import build_architect_pipeline

        p = build_architect_pipeline()
        assert p is not None, "Pipeline should not be None"
        assert len(p.steps) == 1, f"Expected 1 step, got {len(p.steps)}"
        step = p.steps[0]
        step_name = getattr(step, "name", None)
        assert step_name == "GenerateYAML", f"Expected 'GenerateYAML', got '{step_name}'"
    finally:
        # Restore original environment
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
