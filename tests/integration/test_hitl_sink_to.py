"""Integration tests for HITL sink_to functionality (Task 2.1)."""

import pytest
from flujo.domain.dsl import HumanInTheLoopStep, Pipeline, Step
from flujo.domain.models import PipelineContext
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
from tests.conftest import create_test_flujo
from flujo.testing.utils import gather_result
from pydantic import Field


class _Ctx(PipelineContext):
    user_name: str | None = None
    settings: dict[str, str] = Field(default_factory=dict)
    data: str | None = None
    current_item: str | None = None


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_typed_field():
    """Test that sink_to stores response to a typed context field."""
    hitl = HumanInTheLoopStep(
        name="get_user_name", message_for_user="What is your name?", sink_to="user_name"
    )

    pipeline = Pipeline(steps=[hitl])
    runner = create_test_flujo(pipeline, context_model=_Ctx)

    # Run until pause
    paused_result = await gather_result(runner, "initial_input")

    assert paused_result is not None
    assert paused_result.final_pipeline_context.status == "paused"

    # Resume with user response
    resumed = await runner.resume_async(paused_result, "Alice")

    # Verify response was sunk to context
    final_ctx = resumed.final_pipeline_context
    assert final_ctx.user_name == "Alice"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_nested_path():
    """Test that sink_to supports nested paths like settings.theme."""

    # Create a setup step to initialize nested structure
    async def setup_nested(_: str, *, context: PipelineContext | None = None) -> str:
        if isinstance(context, _Ctx):
            context.settings = {}
        return "setup_done"

    setup = Step.from_callable(setup_nested, name="setup", updates_context=True)
    hitl = HumanInTheLoopStep(
        name="get_preference", message_for_user="Select theme:", sink_to="settings.theme"
    )

    pipeline = setup >> hitl
    runner = create_test_flujo(pipeline, context_model=_Ctx)

    # Run until pause
    paused_result = await gather_result(runner, "input")
    assert paused_result is not None

    # Resume
    resumed = await runner.resume_async(paused_result, "dark")
    final_ctx = resumed.final_pipeline_context

    # Verify nested storage (graceful - warning on failure is OK)
    assert final_ctx is not None
    assert final_ctx.settings.get("theme") == "dark"


@pytest.mark.asyncio
async def test_hitl_sink_fails_gracefully_on_invalid_path():
    """Test that invalid sink_to path logs warning but doesn't crash."""
    hitl = HumanInTheLoopStep(
        name="test_invalid", message_for_user="Test", sink_to="nonexistent.deeply.nested.path"
    )

    pipeline = Pipeline(steps=[hitl])
    runner = create_test_flujo(pipeline)

    # Should pause without crashing
    paused_result = await gather_result(runner, "input")
    assert paused_result is not None
    assert paused_result.final_pipeline_context.status == "paused"

    # Should be able to resume (sink failure is graceful - just logs warning)
    resumed = await runner.resume_async(paused_result, "test_value")
    assert resumed is not None


@pytest.mark.asyncio
async def test_hitl_sink_with_updates_context_true():
    """Test that sink_to works when updates_context: true is also set."""
    hitl = HumanInTheLoopStep(
        name="get_data",
        message_for_user="Enter data:",
        sink_to="data",
        updates_context=True,
    )

    pipeline = Pipeline(steps=[hitl])
    runner = create_test_flujo(pipeline, context_model=_Ctx)

    # Run until pause
    paused_result = await gather_result(runner, "input")
    assert paused_result is not None

    # Resume and verify both context update and sink work
    resumed = await runner.resume_async(paused_result, "test_value")
    final_ctx = resumed.final_pipeline_context

    # Both mechanisms should work together
    assert final_ctx is not None
    assert final_ctx.data == "test_value"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_in_loop_iterations():
    """Test that sink_to works correctly in loop iterations."""
    yaml_content = """
version: "0.1"
steps:
  - kind: loop
    name: collect_loop
    loop:
      max_loops: 2
      body:
        - kind: hitl
          name: get_item
          message: "Enter item:"
          sink_to: "current_item"
      exit_expression: "true"  # Exit after first iteration
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    runner = create_test_flujo(pipeline, context_model=_Ctx)

    # First pause
    paused = await gather_result(runner, "input")
    assert paused is not None
    assert paused.final_pipeline_context.status == "paused"

    # Resume - should complete since exit_expression is true
    final = await runner.resume_async(paused, "item_1")
    assert final is not None
    # Verify sink worked
    assert final.final_pipeline_context.current_item == "item_1"


@pytest.mark.asyncio
async def test_hitl_yaml_with_sink_to():
    """Test that YAML blueprint correctly parses and compiles sink_to."""
    yaml_content = """
version: "0.1"
steps:
  - kind: hitl
    name: user_input
    message: "Enter your name:"
    sink_to: "user_name"
  
  - kind: step
    name: greet
    agent: { id: "flujo.builtins.passthrough" }
    input: "Hello {{ context.user_name }}!"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)

    # Verify pipeline loaded correctly
    assert len(pipeline.steps) == 2

    hitl_step = pipeline.steps[0]
    assert isinstance(hitl_step, HumanInTheLoopStep)
    assert hitl_step.sink_to == "user_name"
    assert hitl_step.message_for_user == "Enter your name:"
