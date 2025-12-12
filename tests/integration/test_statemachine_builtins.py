"""Integration tests for builtin skills in StateMachine and other step types.

This test suite validates that builtin skills (flujo.builtins.*) work consistently
across all step types (StateMachine, conditional, loop, top-level) and with both
'agent.params' and 'input' parameter syntaxes.
"""

import pytest
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo


class TestBuiltinSkillConsistency:
    """Test that builtin skills work consistently across step types."""

    @pytest.mark.serial  # StateMachine tests have race conditions under heavy xdist load
    async def test_context_merge_in_statemachine_with_params(self) -> None:
        """Test context_merge with agent.params in StateMachine."""
        yaml_content = """
version: "0.1"
name: "test_sm_params"

steps:
  - kind: StateMachine
    name: test_sm
    start_state: init
    end_states: [complete]
    
    states:
      init:
        steps:
          - kind: step
            name: set_value
            agent:
              id: "flujo.builtins.context_merge"
              params:
                path: "import_artifacts"
                value: { test_key: "params_value" }
            updates_context: true
          - kind: step
            name: set_next
            agent:
              id: "flujo.builtins.context_set"
              params:
                path: "next_state"
                value: "complete"
            updates_context: true
      
      complete:
        steps:
          - kind: step
            name: done
            agent:
              id: "flujo.builtins.passthrough"
            input: "done"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        assert result.final_pipeline_context.import_artifacts.get("test_key") == "params_value"
        # StateMachine reads next_state and transitions, but doesn't update current_state for terminal states
        assert result.final_pipeline_context.next_state == "complete"

    @pytest.mark.serial  # StateMachine tests have race conditions under heavy xdist load
    async def test_context_merge_in_statemachine_with_input(self) -> None:
        """Test context_merge with input in StateMachine."""
        yaml_content = """
version: "0.1"
name: "test_sm_input"

steps:
  - kind: StateMachine
    name: test_sm
    start_state: init
    end_states: [complete]
    
    states:
      init:
        steps:
          - kind: step
            name: set_value
            agent:
              id: "flujo.builtins.context_merge"
            input:
              path: "import_artifacts"
              value: { test_key: "input_value" }
            updates_context: true
          - kind: step
            name: set_next
            agent:
              id: "flujo.builtins.context_set"
            input:
              path: "next_state"
              value: "complete"
            updates_context: true
      
      complete:
        steps:
          - kind: step
            name: done
            agent:
              id: "flujo.builtins.passthrough"
            input: "done"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        assert result.final_pipeline_context.import_artifacts.get("test_key") == "input_value"
        # StateMachine reads next_state and transitions, but doesn't update current_state for terminal states
        assert result.final_pipeline_context.next_state == "complete"

    @pytest.mark.fast
    async def test_context_merge_in_toplevel_with_params(self) -> None:
        """Test context_merge with agent.params in top-level steps."""
        yaml_content = """
version: "0.1"
name: "test_toplevel_params"

steps:
  - kind: step
    name: set_value
    agent:
      id: "flujo.builtins.context_merge"
      params:
        path: "import_artifacts"
        value: { test_key: "toplevel_params" }
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        # Check that context_merge succeeded
        assert result.step_history[0].success
        # Check that context was updated
        assert result.final_pipeline_context.import_artifacts.get("test_key") == "toplevel_params"

    @pytest.mark.fast
    async def test_context_merge_in_toplevel_with_input(self) -> None:
        """Test context_merge with input in top-level steps."""
        yaml_content = """
version: "0.1"
name: "test_toplevel_input"

steps:
  - kind: step
    name: set_value
    agent:
      id: "flujo.builtins.context_merge"
    input:
      path: "import_artifacts"
      value: { test_key: "toplevel_input" }
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        # Check that context_merge succeeded
        assert result.step_history[0].success
        # Check that context was updated
        assert result.final_pipeline_context.import_artifacts.get("test_key") == "toplevel_input"

    @pytest.mark.fast
    async def test_context_merge_in_conditional_branch(self) -> None:
        """Test context_merge in conditional branch."""
        yaml_content = """
version: "0.1"
name: "test_conditional"

steps:
  - kind: conditional
    name: branch_test
    condition_expression: "'yes'"
    branches:
      "yes":
        - kind: step
          name: set_value
          agent:
            id: "flujo.builtins.context_merge"
            params:
              path: "import_artifacts"
              value: { branch_key: "yes_branch" }
          updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        # Check that context was updated in the branch
        assert result.final_pipeline_context.import_artifacts.get("branch_key") == "yes_branch"

    @pytest.mark.fast
    async def test_context_set_with_params(self) -> None:
        """Test context_set builtin with params."""
        yaml_content = """
version: "0.1"
name: "test_context_set"

steps:
  - kind: step
    name: set_counter
    agent:
      id: "flujo.builtins.context_set"
      params:
        path: "import_artifacts.counter"
        value: 42
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        assert result.step_history[0].success
        assert result.final_pipeline_context.import_artifacts.get("counter") == 42

    @pytest.mark.fast
    async def test_context_set_with_input(self) -> None:
        """Test context_set builtin with input."""
        yaml_content = """
version: "0.1"
name: "test_context_set_input"

steps:
  - kind: step
    name: set_counter
    agent:
      id: "flujo.builtins.context_set"
    input:
      path: "import_artifacts.counter"
      value: 99
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        assert result.step_history[0].success
        assert result.final_pipeline_context.import_artifacts.get("counter") == 99

    @pytest.mark.serial  # StateMachine tests have race conditions under heavy xdist load
    async def test_statemachine_dynamic_transitions(self) -> None:
        """Test StateMachine with dynamic next_state transitions."""
        yaml_content = """
version: "0.1"
name: "test_dynamic_transitions"

steps:
  - kind: StateMachine
    name: test_sm
    start_state: init
    end_states: [final]
    
    states:
      init:
        steps:
          - kind: step
            name: goto_middle
            agent:
              id: "flujo.builtins.context_merge"
            input:
              path: "import_artifacts"
              value: { step_count: 1 }
            updates_context: true
          - kind: step
            name: set_next_middle
            agent:
              id: "flujo.builtins.context_set"
              params:
                path: "next_state"
                value: "middle"
            updates_context: true
      
      middle:
        steps:
          - kind: step
            name: goto_final
            agent:
              id: "flujo.builtins.context_merge"
              params:
                path: "import_artifacts"
                value: {}
            updates_context: true
          - kind: step
            name: set_next_final
            agent:
              id: "flujo.builtins.context_set"
              params:
                path: "next_state"
                value: "final"
            updates_context: true
      
      final:
        steps:
          - kind: step
            name: done
            agent:
              id: "flujo.builtins.passthrough"
            input: "reached_final"
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")

        assert result.success
        # Verify the state machine transitioned through init -> middle -> final
        # The context should show we set next_state to trigger transitions
        assert result.final_pipeline_context.next_state == "final"
        assert result.final_pipeline_context.import_artifacts.get("step_count") == 1
        assert result.step_history[-1].name == "test_sm"
