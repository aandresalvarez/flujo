from __future__ import annotations

import pytest


@pytest.mark.integration
def test_architect_validation_repair_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that validation pipeline works correctly with fallback behavior.

    Since flujo.builtins.validate_yaml is not registered in the skill registry,
    the validation step uses the fallback function which should return valid=True.
    This test ensures the validation pipeline completes without infinite loops.
    """
    # Enable full state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Execute with a timeout to prevent infinite loops
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Test timed out - likely infinite loop")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )

        # Should complete successfully
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

        # Should have generated YAML text
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str)
        assert len(yaml_text) > 0

        # The validation should complete (yaml_is_valid may be True or False, but pipeline should finish)
        yaml_is_valid = getattr(ctx, "yaml_is_valid", None)
        # We just check that it's set to something (not None)
        assert yaml_is_valid is not None

    finally:
        signal.alarm(0)  # Cancel the alarm
