from __future__ import annotations

import pytest
from typing import Any


class StubAgent:
    """Simple stub agent that returns predictable outputs for testing."""

    def __init__(self, return_value: Any):
        self.return_value = return_value

    async def __call__(self, context: Any) -> Any:
        return self.return_value


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to architect pipeline execution
async def test_architect_state_machine_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Architect completes the full state machine flow successfully."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner

    # Build the architect pipeline
    pipeline = build_architect_pipeline()

    # Verify pipeline was built
    assert pipeline is not None, "Pipeline should be built successfully"
    assert hasattr(pipeline, "steps"), "Pipeline should have steps"

    # Create runner with ArchitectContext
    initial = {
        "initial_prompt": "Create a data processing pipeline",
        "user_goal": "Process CSV data and generate reports",
    }
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Verify runner was created
    assert runner is not None, "Runner should be created successfully"

    # Run the pipeline to completion
    result = None
    iteration_count = 0
    max_iterations = 20

    async for item in runner.run_async("Create a data processing pipeline"):
        result = item
        iteration_count += 1
        if iteration_count >= max_iterations:
            print(f"DEBUG: Stopping after {max_iterations} iterations")
            break

    # Verify the result
    assert result is not None, "Pipeline should produce some result"
    assert hasattr(result, "step_history"), "Result should have step history"

    # Check that steps executed
    step_history = getattr(result, "step_history", []) or []
    assert len(step_history) > 0, "At least some steps should have executed"

    print(f"DEBUG: Pipeline completed in {iteration_count} iterations")
    print(f"DEBUG: Executed {len(step_history)} top-level steps")

    # Verify the pipeline context was created
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None, "Pipeline context should be available"

    # Check that YAML was generated
    yaml_text = getattr(ctx, "yaml_text", None)
    assert yaml_text is not None, "YAML text should be generated"
    assert isinstance(yaml_text, str), "YAML text should be a string"
    assert len(yaml_text) > 0, "YAML text should not be empty"

    print(f"DEBUG: Generated YAML length: {len(yaml_text)}")
    print("DEBUG: Architect state machine completed successfully")


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to multiple architect pipeline runs
async def test_architect_different_goal_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Architect handles different types of goals correctly."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner

    # Build the architect pipeline
    pipeline = build_architect_pipeline()

    # Test different goal types
    test_goals = [
        "Create a web scraping pipeline",
        "Build a machine learning model",
        "Set up a CI/CD pipeline",
        "Create a REST API",
        "Build a data visualization dashboard",
    ]

    for goal in test_goals:
        print(f"DEBUG: Testing goal: {goal}")

        # Create runner with different goal
        initial = {"initial_prompt": goal, "user_goal": goal}
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        # Run the pipeline
        result = None
        iteration_count = 0
        max_iterations = 15

        async for item in runner.run_async(goal):
            result = item
            iteration_count += 1
            if iteration_count >= max_iterations:
                break

        # Verify the result
        assert result is not None, f"Pipeline should produce result for goal: {goal}"
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None, f"Pipeline context should be available for goal: {goal}"

        # Check that YAML was generated
        yaml_text = getattr(ctx, "yaml_text", None)
        assert yaml_text is not None, f"YAML text should be generated for goal: {goal}"
        assert isinstance(yaml_text, str), f"YAML text should be a string for goal: {goal}"
        assert len(yaml_text) > 0, f"YAML text should not be empty for goal: {goal}"

        print(f"DEBUG: Goal '{goal}' completed successfully in {iteration_count} iterations")

    print("DEBUG: All goal types completed successfully")


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to architect pipeline execution
async def test_architect_context_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Architect maintains context throughout the pipeline execution."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner

    # Build the architect pipeline
    pipeline = build_architect_pipeline()

    # Create runner with specific context
    initial = {
        "initial_prompt": "Create a pipeline with specific requirements",
        "user_goal": "Build a secure data processing pipeline",
        "non_interactive": True,  # Test non-interactive mode
        "hitl_enabled": False,  # Test HITL disabled
    }
    runner = create_flujo_runner(
        pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
    )

    # Run the pipeline
    result = None
    iteration_count = 0
    max_iterations = 15

    async for item in runner.run_async("Create a pipeline with specific requirements"):
        result = item
        iteration_count += 1
        if iteration_count >= max_iterations:
            break

    # Verify the result
    assert result is not None, "Pipeline should produce some result"
    ctx = getattr(result, "final_pipeline_context", None)
    assert ctx is not None, "Pipeline context should be available"

    # Check that initial context was preserved
    assert getattr(ctx, "initial_prompt", None) == "Create a pipeline with specific requirements", (
        "Initial prompt should be preserved"
    )
    assert getattr(ctx, "user_goal", None) == "Build a secure data processing pipeline", (
        "User goal should be preserved"
    )
    assert getattr(ctx, "non_interactive", None) is True, "Non-interactive flag should be preserved"
    assert getattr(ctx, "hitl_enabled", None) is False, "HITL enabled flag should be preserved"

    # Check that YAML was generated
    yaml_text = getattr(ctx, "yaml_text", None)
    assert yaml_text is not None, "YAML text should be generated"

    print("DEBUG: Context persistence test completed successfully")


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to architect pipeline execution with error handling
async def test_architect_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Architect handles errors gracefully and falls back to reliable defaults."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner
    from flujo.infra.skill_registry import get_skill_registry

    # Setup: Mock the skill registry to simulate errors
    registry = get_skill_registry()

    # Mock the skill registry to return errors for skills that would normally be used
    def mock_registry_get(skill_id: str):
        # Simulate errors for certain skills
        if "error" in skill_id:
            raise Exception("Simulated skill error")
        # Return empty results to force fallback usage
        return {}

    # Store original get method and replace it
    registry._get_original = registry.get
    registry.get = mock_registry_get

    try:
        # Build the architect pipeline
        pipeline = build_architect_pipeline()

        # Create runner with ArchitectContext
        initial = {
            "initial_prompt": "Create a pipeline that might encounter errors",
            "user_goal": "Build a robust pipeline",
        }
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        # Run the pipeline
        result = None
        iteration_count = 0
        max_iterations = 15

        async for item in runner.run_async("Create a pipeline that might encounter errors"):
            result = item
            iteration_count += 1
            if iteration_count >= max_iterations:
                break

        # Verify the result - the Architect should handle errors gracefully
        assert result is not None, "Pipeline should produce result even with errors"
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None, "Pipeline context should be available even with errors"

        # Check that YAML was generated despite errors
        yaml_text = getattr(ctx, "yaml_text", None)
        assert yaml_text is not None, "YAML text should be generated even with errors"
        assert isinstance(yaml_text, str), "YAML text should be a string even with errors"
        assert len(yaml_text) > 0, "YAML text should not be empty even with errors"

        print("DEBUG: Error handling test completed successfully")
        print("DEBUG: Architect demonstrated robustness by falling back to reliable defaults")

    finally:
        # Restore original registry method
        if hasattr(registry, "_get_original"):
            registry.get = registry._get_original
            delattr(registry, "_get_original")


@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow due to multiple architect pipeline runs
async def test_architect_performance_characteristics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the Architect performs consistently across multiple runs."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner
    import time

    # Build the architect pipeline
    pipeline = build_architect_pipeline()

    # Test multiple runs to check consistency
    run_times = []
    yaml_lengths = []

    for run_num in range(3):
        print(f"DEBUG: Performance test run {run_num + 1}")

        start_time = time.time()

        # Create runner
        initial = {
            "initial_prompt": f"Create pipeline run {run_num + 1}",
            "user_goal": "Build a consistent pipeline",
        }
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        # Run the pipeline
        result = None
        iteration_count = 0
        max_iterations = 15

        async for item in runner.run_async(f"Create pipeline run {run_num + 1}"):
            result = item
            iteration_count += 1
            if iteration_count >= max_iterations:
                break

        end_time = time.time()
        run_time = end_time - start_time

        # Verify the result
        assert result is not None, f"Pipeline should produce result for run {run_num + 1}"
        ctx = getattr(result, "final_pipeline_context", None)
        assert ctx is not None, f"Pipeline context should be available for run {run_num + 1}"

        # Check that YAML was generated
        yaml_text = getattr(ctx, "yaml_text", None)
        assert yaml_text is not None, f"YAML text should be generated for run {run_num + 1}"

        # Record metrics
        run_times.append(run_time)
        yaml_lengths.append(len(yaml_text))

        print(
            f"DEBUG: Run {run_num + 1} completed in {run_time:.2f}s with {iteration_count} iterations"
        )

    # Verify consistency
    assert len(set(yaml_lengths)) <= 2, (
        f"YAML lengths should be consistent across runs, got: {yaml_lengths}"
    )
    assert max(run_times) - min(run_times) < 5.0, (
        f"Run times should be consistent, got: {run_times}"
    )

    print("DEBUG: Performance test completed successfully")
    print(f"DEBUG: Run times: {[f'{t:.2f}s' for t in run_times]}")
    print(f"DEBUG: YAML lengths: {yaml_lengths}")
    print("DEBUG: Architect demonstrated consistent performance across multiple runs")
