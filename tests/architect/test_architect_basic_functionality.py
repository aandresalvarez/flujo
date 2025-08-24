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
async def test_architect_basic_functionality(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test basic Architect functionality - pipeline builds and runs."""

    # Enable state machine
    monkeypatch.setenv("FLUJO_ARCHITECT_STATE_MACHINE", "1")

    from flujo.architect.builder import build_architect_pipeline
    from flujo.architect.context import ArchitectContext
    from flujo.cli.helpers import create_flujo_runner
    from flujo.infra.skill_registry import get_skill_registry

    # Setup: Replace powerful LLM agents with StubAgents
    registry = get_skill_registry()

    # Mock goal_decomposer to return a fixed list of tasks
    goal_decomposer_stub = StubAgent(
        {"decomposed_tasks": ["Fetch web page content", "Process the content", "Return results"]}
    )

    # Mock yaml_writer to return predictable YAML
    yaml_writer_stub = StubAgent(
        {
            "yaml_content": """version: "0.1"
steps:
- name: FetchWebPage
  agent:
    id: flujo.builtins.http_get
    params:
      url: "https://example.com"
- name: ProcessContent
  agent:
    id: flujo.builtins.stringify
    params: {}
- name: ReturnResults
  agent:
    id: flujo.builtins.return_value
    params: {}"""
        }
    )

    # Mock the built-in HITL skills to automatically return "approved"
    ask_user_stub = StubAgent("Y")  # User responds "Y" (yes)
    check_user_confirmation_stub = StubAgent("approved")  # Confirmation check returns "approved"

    # Mock the skills in the registry
    def mock_registry_get(skill_id: str):
        if skill_id == "flujo.architect.goal_decomposer":
            return {"factory": lambda: goal_decomposer_stub}
        elif skill_id == "flujo.architect.yaml_writer":
            return {"factory": lambda: yaml_writer_stub}
        elif skill_id == "flujo.builtins.ask_user":
            return {"factory": lambda: ask_user_stub}
        elif skill_id == "flujo.builtins.check_user_confirmation":
            return {"factory": lambda: check_user_confirmation_stub}
        else:
            return registry._get_original(skill_id)

    # Store original get method and replace it
    registry._get_original = registry.get
    registry.get = mock_registry_get

    try:
        # Build the architect pipeline
        pipeline = build_architect_pipeline()

        # Verify pipeline was built
        assert pipeline is not None, "Pipeline should be built successfully"
        assert hasattr(pipeline, "steps"), "Pipeline should have steps"

        # Create runner with ArchitectContext
        initial = {
            "initial_prompt": "Create a web scraping pipeline",
            "user_goal": "Fetch and process web content",
        }
        runner = create_flujo_runner(
            pipeline=pipeline, context_model_class=ArchitectContext, initial_context_data=initial
        )

        # Verify runner was created
        assert runner is not None, "Runner should be created successfully"

        # Try to run the pipeline (but limit iterations to avoid infinite loops)
        result = None
        iteration_count = 0
        max_iterations = 10

        async for item in runner.run_async("Create a web scraping pipeline"):
            result = item
            iteration_count += 1
            if iteration_count >= max_iterations:
                print(f"DEBUG: Stopping after {max_iterations} iterations to avoid infinite loop")
                break

        # Basic verification that something happened
        assert result is not None, "Pipeline should produce some result"
        assert hasattr(result, "step_history"), "Result should have step history"

        # Check that some steps executed
        step_history = getattr(result, "step_history", []) or []
        assert len(step_history) > 0, "At least some steps should have executed"

        print(f"DEBUG: Pipeline executed {len(step_history)} top-level steps")
        print(f"DEBUG: Step names: {[getattr(step, 'name', 'unknown') for step in step_history]}")

        # Verify the pipeline context was created
        ctx = getattr(result, "final_pipeline_context", None)
        if ctx is not None:
            print(
                f"DEBUG: Context has attributes: {[attr for attr in dir(ctx) if not attr.startswith('_')]}"
            )

            # Check if any YAML was generated
            yaml_text = getattr(ctx, "yaml_text", None)
            if yaml_text:
                print(f"DEBUG: YAML text length: {len(yaml_text)}")
                print(f"DEBUG: YAML preview: {yaml_text[:200]}...")

    finally:
        # Restore original registry method
        if hasattr(registry, "_get_original"):
            registry.get = registry._get_original
            delattr(registry, "_get_original")
