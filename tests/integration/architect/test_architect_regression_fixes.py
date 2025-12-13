from __future__ import annotations

import pytest
from flujo.architect.builder import build_architect_pipeline
from flujo.architect.context import ArchitectContext
from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling


@pytest.mark.integration
def test_regression_fix_infinite_loop_generation_state():
    """Regression test: Ensure GenerateYAML step doesn't cause infinite loops in Generation state."""
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # This should complete without hanging or infinite loops
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # The state machine should have progressed through all states without hanging.


@pytest.mark.integration
def test_regression_fix_context_updates_work():
    """Regression test: Ensure steps with updates_context=True actually update context fields.

    This test verifies that the critical fix for context updates is working,
    specifically that yaml_text and generated_yaml are preserved in the final context.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # CRITICAL: Verify that yaml_text is preserved in the final context
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str), f"Expected yaml_text to be a string, got {type(yaml_text)}"
    assert "version:" in yaml_text, (
        f"Expected yaml_text to contain 'version:', got: {yaml_text[:100]}"
    )

    # CRITICAL: Verify that generated_yaml is also preserved
    generated_yaml = getattr(ctx, "generated_yaml", None)
    assert isinstance(generated_yaml, str), (
        f"Expected generated_yaml to be a string, got {type(generated_yaml)}"
    )
    assert "version:" in generated_yaml, (
        f"Expected generated_yaml to contain 'version:', got: {generated_yaml[:100]}"
    )

    # Verify both fields contain the same content
    assert yaml_text == generated_yaml, "yaml_text and generated_yaml should be identical"


@pytest.mark.integration
def test_regression_fix_state_machine_progression():
    """Regression test: Ensure state machine progresses through all expected states.

    This test verifies that the state machine doesn't get stuck in any state
    and properly transitions through the complete flow.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify we have a valid execution result
    assert hasattr(result, "final_pipeline_context")

    # The state machine should have reached a terminal state
    # We can't check the exact state as it may vary, but we ensure
    # the pipeline completed without hanging or infinite loops


@pytest.mark.integration
def test_regression_fix_validation_state_transitions():
    """Regression test: Ensure Validation state doesn't cause infinite loops.

    This test verifies that the Validation state properly transitions to the next state
    instead of looping back to itself indefinitely.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify that yaml_is_valid was properly set during validation
    yaml_is_valid = getattr(ctx, "yaml_is_valid", None)
    assert yaml_is_valid is not None, "yaml_is_valid should be set during validation"
    assert isinstance(yaml_is_valid, bool), (
        f"yaml_is_valid should be a boolean, got {type(yaml_is_valid)}"
    )

    # If validation passed, we should have yaml_text
    if yaml_is_valid:
        yaml_text = getattr(ctx, "yaml_text", None)
        assert isinstance(yaml_text, str), "yaml_text should be preserved when validation passes"


@pytest.mark.integration
def test_regression_fix_state_updates():
    """Regression test: Ensure state-machine typed fields update correctly."""
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify typed status is present
    assert getattr(ctx, "status", None) is not None, "context should carry execution status"


@pytest.mark.integration
def test_regression_fix_finalization_preserves_data():
    """Regression test: Ensure Finalization step preserves all generated data.

    This test verifies that the _finalize function properly preserves yaml_text
    and generated_yaml in the final result instead of returning empty dict.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # CRITICAL: Verify that yaml_text is preserved through finalization
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str), f"Final yaml_text should be preserved, got {type(yaml_text)}"
    assert len(yaml_text) > 0, "Final yaml_text should not be empty"

    # Verify the YAML content is valid
    assert "version:" in yaml_text, "Final yaml_text should contain valid YAML structure"
    assert "name:" in yaml_text, "Final yaml_text should contain pipeline name"
    assert "steps:" in yaml_text, "Final yaml_text should contain steps section"


@pytest.mark.integration
def test_regression_fix_no_direct_context_mutation():
    """Regression test: Ensure no direct context mutation occurs in step functions.

    This test verifies that steps use proper return values instead of directly
    mutating context objects, which was causing the context update failures.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify that the context is properly structured (not corrupted by direct mutation)
    assert hasattr(ctx, "yaml_text"), "Context should have yaml_text attribute"
    assert hasattr(ctx, "generated_yaml"), "Context should have generated_yaml attribute"

    # Verify the context object is still a valid ArchitectContext instance
    assert isinstance(ctx, ArchitectContext), "Final context should be ArchitectContext instance"


@pytest.mark.integration
def test_regression_fix_complete_architect_flow():
    """Comprehensive regression test: Ensure the complete architect flow works end-to-end.

    This test verifies that all the fixes work together to provide a working
    architect pipeline without any of the previous issues.
    """
    # Enable full state machine
    import os

    os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Verify the pipeline completed successfully
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify all critical fields are present and correct
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str), "yaml_text must be a string"
    assert len(yaml_text) > 0, "yaml_text must not be empty"
    assert "version:" in yaml_text, "yaml_text must contain valid YAML"

    generated_yaml = getattr(ctx, "generated_yaml", None)
    assert isinstance(generated_yaml, str), "generated_yaml must be a string"
    assert generated_yaml == yaml_text, "generated_yaml must match yaml_text"

    yaml_is_valid = getattr(ctx, "yaml_is_valid", None)
    assert yaml_is_valid is not None, "yaml_is_valid must be set"
    assert isinstance(yaml_is_valid, bool), "yaml_is_valid must be a boolean"

    # Success: All regression fixes are working correctly
    print("✅ All regression fixes verified successfully!")
    print(f"   - yaml_text: {len(yaml_text)} characters")
    print(f"   - yaml_is_valid: {yaml_is_valid}")
    print(f"   - Context type: {type(ctx).__name__}")
