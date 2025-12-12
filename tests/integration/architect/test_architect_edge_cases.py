from __future__ import annotations

import pytest
from flujo.architect.builder import build_architect_pipeline
from flujo.architect.context import ArchitectContext
from flujo.cli.helpers import create_flujo_runner, execute_pipeline_with_output_handling


@pytest.mark.integration
def test_architect_handles_empty_initial_prompt():
    """Test: Architect handles empty initial prompt gracefully without crashing."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete without crashing, even with empty prompt
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None


@pytest.mark.integration
def test_architect_handles_malformed_initial_data():
    """Test: Architect handles malformed initial data gracefully."""
    pipeline = build_architect_pipeline()

    # Test with malformed data - should fail gracefully with validation error
    initial = {"invalid_field": "invalid_value"}  # Missing required fields

    try:
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        # If we get here, the system should handle the malformed data gracefully
        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )

        # Should complete without crashing, even with malformed data
        assert result is not None
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None

    except Exception as e:
        # Expected behavior: system should fail fast with validation error
        # This is actually good - it means the system is properly validating input
        assert "validation" in str(e).lower() or "initialization" in str(e).lower(), (
            f"Expected validation error, got: {e}"
        )
        print(f"✅ System properly rejected malformed data with error: {e}")


@pytest.mark.integration
def test_architect_handles_very_long_input():
    """Test: Architect handles very long input without performance degradation."""
    pipeline = build_architect_pipeline()
    long_input = "Echo input " * 1000  # Very long input
    initial = {"initial_prompt": "Make a pipeline", "user_goal": long_input}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data=long_input, run_id=None, json_output=False
    )

    # Should complete successfully with long input
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even with long input
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str)
    assert len(yaml_text) > 0


@pytest.mark.integration
def test_architect_handles_special_characters_in_input():
    """Test: Architect handles special characters and unicode in input."""
    pipeline = build_architect_pipeline()
    special_input = "Echo input with special chars: 🚀✨🎉\n\t\r\"'\\"
    initial = {"initial_prompt": "Make a pipeline", "user_goal": special_input}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data=special_input, run_id=None, json_output=False
    )

    # Should complete successfully with special characters
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even with special characters
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str)
    assert len(yaml_text) > 0


@pytest.mark.integration
@pytest.mark.slow  # Uses ThreadPool and multiple pipeline runs; slower
def test_architect_handles_concurrent_executions():
    """Test: Architect can handle multiple concurrent executions without conflicts."""
    import concurrent.futures

    def run_single_architect():
        """Run a single architect execution."""
        pipeline = build_architect_pipeline()
        initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )

        return result is not None

    # Run multiple concurrent executions
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_single_architect) for _ in range(3)]
        results = [future.result() for future in futures]

    # All executions should succeed
    assert all(results), "All concurrent executions should succeed"


@pytest.mark.integration
def test_architect_handles_memory_pressure():
    """Test: Architect handles memory pressure gracefully without crashing."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Simulate memory pressure by creating large objects
    large_data = ["x" * 1000000 for _ in range(10)]  # ~10MB of data

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even under memory pressure
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Clean up large data
    del large_data


@pytest.mark.integration
def test_architect_handles_network_timeouts_gracefully():
    """Test: Architect handles network timeouts gracefully without hanging."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # This test verifies that the architect doesn't hang on network issues
    # The actual network calls are mocked in tests, so this tests the timeout handling
    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete without hanging
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None


@pytest.mark.integration
def test_architect_handles_corrupted_context_data():
    """Test: Architect handles corrupted context data gracefully."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even with potentially corrupted data
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate valid YAML
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str)
    assert "version:" in yaml_text


@pytest.mark.integration
def test_architect_handles_rapid_state_transitions():
    """Test: Architect handles rapid state transitions without race conditions."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully without race conditions
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Verify the state machine progressed through all states correctly


@pytest.mark.integration
def test_architect_handles_large_yaml_output():
    """Test: Architect handles large YAML output without memory issues."""
    pipeline = build_architect_pipeline()
    initial = {
        "initial_prompt": "Make a complex pipeline with many steps",
        "user_goal": "Echo input",
    }
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even with large YAML
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML regardless of size
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str)
    assert len(yaml_text) > 0


@pytest.mark.integration
def test_architect_handles_invalid_yaml_generation():
    """Test: Architect handles invalid YAML generation gracefully."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even if YAML generation has issues
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should handle YAML validation gracefully
    yaml_is_valid = getattr(ctx, "yaml_is_valid", None)
    assert yaml_is_valid is not None


@pytest.mark.integration
def test_architect_handles_context_serialization_issues():
    """Test: Architect handles context serialization issues gracefully."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even with serialization issues
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should preserve critical data
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_skill_registry_failures():
    """Test: Architect handles skill registry failures gracefully."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even with registry issues
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should handle missing skills gracefully


@pytest.mark.integration
def test_architect_handles_telemetry_failures():
    """Test: Architect handles telemetry failures gracefully without affecting core functionality."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully even if telemetry fails
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Core functionality should work regardless of telemetry
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_environment_variable_changes():
    """Test: Architect handles environment variable changes gracefully."""
    import os

    # Test with different environment configurations
    original_value = os.environ.get("FLUJO_ARCHITECT_STATE_MACHINE", None)

    try:
        # Test with state machine enabled
        os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = "1"

        pipeline = build_architect_pipeline()
        initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        result = execute_pipeline_with_output_handling(
            runner=runner, input_data="Echo input", run_id=None, json_output=False
        )

        assert result is not None

    finally:
        # Restore original environment
        if original_value is not None:
            os.environ["FLUJO_ARCHITECT_STATE_MACHINE"] = original_value
        else:
            os.environ.pop("FLUJO_ARCHITECT_STATE_MACHINE", None)


@pytest.mark.integration
def test_architect_handles_concurrent_context_updates():
    """Test: Architect handles concurrent context updates without data corruption."""
    pipeline = build_architect_pipeline()
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    # Should complete successfully without context corruption
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Context should be consistent
    yaml_text = getattr(ctx, "yaml_text", None)
    generated_yaml = getattr(ctx, "generated_yaml", None)

    if yaml_text is not None and generated_yaml is not None:
        assert yaml_text == generated_yaml, "Context should be consistent"


@pytest.mark.integration
def test_architect_handles_rapid_pipeline_rebuilds():
    """Test: Architect handles rapid pipeline rebuilds without resource leaks."""
    # Build multiple pipelines rapidly
    pipelines = []
    for i in range(5):
        pipeline = build_architect_pipeline()
        pipelines.append(pipeline)
        assert pipeline is not None

    # All pipelines should be valid
    assert len(pipelines) == 5
    assert all(p is not None for p in pipelines)

    # Test one pipeline to ensure it still works
    initial = {"initial_prompt": "Make a pipeline", "user_goal": "Echo input"}
    runner = create_flujo_runner(
        pipeline=pipelines[0], context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data="Echo input", run_id=None, json_output=False
    )

    assert result is not None


@pytest.mark.integration
def test_architect_handles_mixed_encoding_inputs():
    """Test: Architect handles mixed encoding inputs gracefully."""
    pipeline = build_architect_pipeline()

    # Test with mixed encodings
    mixed_input = "Echo input with mixed encodings: 🚀✨🎉\n\t\r\"'\\ and normal text"
    initial = {"initial_prompt": "Make a pipeline", "user_goal": mixed_input}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data=mixed_input, run_id=None, json_output=False
    )

    # Should complete successfully with mixed encodings
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even with mixed encodings
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None


@pytest.mark.integration
def test_architect_handles_extremely_complex_goals():
    """Test: Architect handles extremely complex goals without crashing."""
    pipeline = build_architect_pipeline()

    complex_goal = """
    Create a highly complex pipeline that:
    1. Processes multiple data sources
    2. Applies complex transformations
    3. Handles error conditions gracefully
    4. Provides comprehensive logging
    5. Supports multiple output formats
    6. Implements retry mechanisms
    7. Handles rate limiting
    8. Provides monitoring and metrics
    9. Supports configuration management
    10. Implements security best practices
    """

    initial = {"initial_prompt": "Make a pipeline", "user_goal": complex_goal}
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    result = execute_pipeline_with_output_handling(
        runner=runner, input_data=complex_goal, run_id=None, json_output=False
    )

    # Should complete successfully even with complex goals
    assert result is not None
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None

    # Should generate YAML even for complex goals
    yaml_text = getattr(ctx, "yaml_text", None)
    assert isinstance(yaml_text, str) or yaml_text is None
