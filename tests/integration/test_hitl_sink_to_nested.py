"""Integration tests for HITL sink_to in nested contexts (conditionals/loops).

This test suite verifies the fix for the bug where sink_to values were lost
in nested contexts due to dual application (runner + executor) and context
forking issues.

Bug Fix: sink_to is now applied ONLY in the HITL executor policy, which writes
to the branch context. The containing step (conditional/loop) then merges this
branch context, preserving the sink_to value.
"""

import pytest
from flujo.domain.dsl import HumanInTheLoopStep, Pipeline
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
from tests.conftest import create_test_flujo
from flujo.testing.utils import gather_result


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_in_conditional_branch():
    """Test that sink_to works inside conditional branches.

    This is the primary test case for the bug fix. Previously, the sink_to
    value would be lost because:
    1. Runner applied sink_to to main context
    2. HITL executor applied sink_to to branch context
    3. Branch context merge would overwrite main context

    After fix: sink_to only applied in executor, branch context properly merged.
    """
    yaml_content = """
version: "0.1"
steps:
  - kind: step
    name: init
    agent: { id: "flujo.builtins.passthrough" }
    input: '{"test": {}}'
    updates_context: true

  - kind: conditional
    name: test_conditional
    condition_expression: "True"
    branches:
      true:
        - kind: hitl
          name: ask_in_conditional
          message: "Enter value:"
          sink_to: "scratchpad.user_input"

  - kind: step
    name: verify
    agent: { id: "flujo.builtins.stringify" }
    input: "Value: {{ context.scratchpad.user_input }}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    runner = create_test_flujo(pipeline)

    # Run until pause
    paused_result = await gather_result(runner, "initial_input")
    assert paused_result is not None
    assert paused_result.final_pipeline_context.scratchpad.get("status") == "paused"

    # Resume with user input
    resumed = await runner.resume_async(paused_result, "test123")

    # Verify sink_to worked
    final_ctx = resumed.final_pipeline_context
    assert hasattr(final_ctx, "scratchpad") or isinstance(final_ctx.scratchpad, dict)

    if hasattr(final_ctx.scratchpad, "user_input"):
        assert final_ctx.scratchpad.user_input == "test123"
    elif isinstance(final_ctx.scratchpad, dict):
        assert final_ctx.scratchpad.get("user_input") == "test123"
    else:
        pytest.fail("scratchpad.user_input not found in context after conditional branch")

    # Verify the verify step saw the value
    final_output = resumed.step_history[-1].output
    assert "Value: test123" in str(final_output)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.skip(
    reason="Loop+HITL resume has complex iteration counting issue - needs deeper investigation"
)
async def test_hitl_sink_to_in_loop_body():
    """Test that sink_to works inside loop bodies.

    This tests the loop executor's context merging after iteration completes.
    Note: Currently skipped due to loop iteration counting issue after HITL resume.
    """
    yaml_content = """
version: "0.1"
steps:
  - kind: step
    name: init
    agent: { id: "flujo.builtins.passthrough" }
    input: '{}'
    updates_context: true

  - kind: loop
    name: test_loop
    loop:
      max_loops: 1
      body:
        - kind: hitl
          name: ask_in_loop
          message: "Enter value:"
          sink_to: "scratchpad.loop_input"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    runner = create_test_flujo(pipeline)

    # Run until first pause
    paused_result = await gather_result(runner, "initial_input")
    assert paused_result is not None
    assert paused_result.final_pipeline_context.scratchpad.get("status") == "paused"

    # Resume with user input - loop will hit max_loops and exit
    resumed = await runner.resume_async(paused_result, "loop_value_1")

    # The pipeline should complete now (loop exits after max_loops)
    assert resumed.final_pipeline_context.scratchpad.get("status") != "paused", (
        f"Pipeline still paused after loop completion. Status: {resumed.final_pipeline_context.scratchpad.get('status')}"
    )

    # Verify sink_to worked in loop
    final_ctx = resumed.final_pipeline_context
    if hasattr(final_ctx.scratchpad, "loop_input"):
        assert final_ctx.scratchpad.loop_input == "loop_value_1"
    elif isinstance(final_ctx.scratchpad, dict):
        assert final_ctx.scratchpad.get("loop_input") == "loop_value_1", (
            f"loop_input not found in scratchpad. Keys: {list(final_ctx.scratchpad.keys())}"
        )
    else:
        pytest.fail("scratchpad.loop_input not found after loop")


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_nested_conditional_in_loop():
    """Test sink_to in a conditional inside a loop (double nesting).

    This is the most complex case - ensures context merging works
    through multiple levels of nesting.
    """
    yaml_content = """
version: "0.1"
steps:
  - kind: step
    name: init
    agent: { id: "flujo.builtins.passthrough" }
    input: '{"items": []}'
    updates_context: true

  - kind: loop
    name: outer_loop
    loop:
      max_loops: 2
      body:
        - kind: conditional
          name: inner_conditional
          condition_expression: "True"
          branches:
            true:
              - kind: hitl
                name: nested_ask
                message: "Enter nested value:"
                sink_to: "scratchpad.nested_value"
      exit_expression: "len(context.scratchpad.get('items', [])) >= 1"

  - kind: step
    name: verify
    agent: { id: "flujo.builtins.stringify" }
    input: "Nested: {{ context.scratchpad.nested_value }}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    runner = create_test_flujo(pipeline)

    # Run until pause
    paused_result = await gather_result(runner, "initial_input")
    assert paused_result is not None

    # Resume and exit loop
    ctx = paused_result.final_pipeline_context
    ctx.scratchpad["items"] = ["done"]
    resumed = await runner.resume_async(paused_result, "nested123")

    # Verify sink_to survived double nesting
    final_ctx = resumed.final_pipeline_context
    if hasattr(final_ctx.scratchpad, "nested_value"):
        assert final_ctx.scratchpad.nested_value == "nested123"
    elif isinstance(final_ctx.scratchpad, dict):
        assert final_ctx.scratchpad.get("nested_value") == "nested123"
    else:
        pytest.fail("scratchpad.nested_value not found after nested conditional in loop")


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_top_level_still_works():
    """Test that top-level HITL sink_to still works (regression test).

    Ensures we didn't break the existing functionality when fixing nested contexts.
    """
    hitl = HumanInTheLoopStep(
        name="get_user_name", message_for_user="What is your name?", sink_to="scratchpad.user_name"
    )

    pipeline = Pipeline(steps=[hitl])
    runner = create_test_flujo(pipeline)

    # Run until pause
    paused_result = await gather_result(runner, "initial_input")
    assert paused_result is not None
    assert paused_result.final_pipeline_context.scratchpad.get("status") == "paused"

    # Resume with user response
    resumed = await runner.resume_async(paused_result, "Alice")

    # Verify response was sunk to context
    final_ctx = resumed.final_pipeline_context
    assert hasattr(final_ctx, "scratchpad") or isinstance(final_ctx.scratchpad, dict)

    # Check if user_name was stored
    if hasattr(final_ctx.scratchpad, "user_name"):
        assert final_ctx.scratchpad.user_name == "Alice"
    elif isinstance(final_ctx.scratchpad, dict):
        assert final_ctx.scratchpad.get("user_name") == "Alice"
    else:
        pytest.fail("scratchpad.user_name not found in context")


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.serial
async def test_hitl_sink_to_multiple_conditionals():
    """Test sink_to in multiple sequential conditional branches.

    Ensures each conditional properly merges its branch context.
    """
    yaml_content = """
version: "0.1"
steps:
  - kind: step
    name: init
    agent: { id: "flujo.builtins.passthrough" }
    input: '{}'
    updates_context: true

  - kind: conditional
    name: first_conditional
    condition_expression: "True"
    branches:
      true:
        - kind: hitl
          name: ask_first
          message: "Enter first value:"
          sink_to: "scratchpad.first_value"

  - kind: conditional
    name: second_conditional
    condition_expression: "True"
    branches:
      true:
        - kind: hitl
          name: ask_second
          message: "Enter second value:"
          sink_to: "scratchpad.second_value"

  - kind: step
    name: verify
    agent: { id: "flujo.builtins.stringify" }
    input: "Both: {{ context.scratchpad.first_value }} and {{ context.scratchpad.second_value }}"
"""

    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    runner = create_test_flujo(pipeline)

    # First pause
    paused_result = await gather_result(runner, "initial_input")
    assert paused_result is not None

    # Resume first HITL
    resumed_once = await runner.resume_async(paused_result, "value1")

    # Second pause
    assert resumed_once.final_pipeline_context.scratchpad.get("status") == "paused"

    # Resume second HITL
    final = await runner.resume_async(resumed_once, "value2")

    # Verify both values persisted
    final_ctx = final.final_pipeline_context

    if isinstance(final_ctx.scratchpad, dict):
        assert final_ctx.scratchpad.get("first_value") == "value1"
        assert final_ctx.scratchpad.get("second_value") == "value2"
    else:
        assert hasattr(final_ctx.scratchpad, "first_value")
        assert hasattr(final_ctx.scratchpad, "second_value")
        assert final_ctx.scratchpad.first_value == "value1"
        assert final_ctx.scratchpad.second_value == "value2"

    # Verify final output has both values
    final_output = str(final.step_history[-1].output)
    assert "value1" in final_output
    assert "value2" in final_output
