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
    
    @pytest.mark.fast
    async def test_context_merge_in_statemachine_with_params(self, tmp_path):
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
                path: "scratchpad"
                value: { test_key: "params_value", next_state: "complete" }
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
        assert result.context.scratchpad.get("test_key") == "params_value"
        assert result.context.scratchpad.get("current_state") == "complete"
    
    @pytest.mark.fast
    async def test_context_merge_in_statemachine_with_input(self, tmp_path):
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
              path: "scratchpad"
              value: { test_key: "input_value", next_state: "complete" }
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
        assert result.context.scratchpad.get("test_key") == "input_value"
        assert result.context.scratchpad.get("current_state") == "complete"
    
    @pytest.mark.fast
    async def test_context_merge_in_toplevel_with_params(self, tmp_path):
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
        path: "scratchpad"
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
        assert result.context.scratchpad.get("test_key") == "toplevel_params"
    
    @pytest.mark.fast
    async def test_context_merge_in_toplevel_with_input(self, tmp_path):
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
      path: "scratchpad"
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
        assert result.context.scratchpad.get("test_key") == "toplevel_input"
    
    @pytest.mark.fast
    async def test_context_merge_in_conditional_branch(self, tmp_path):
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
              path: "scratchpad"
              value: { branch_key: "yes_branch" }
          updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")
        
        assert result.success
        # Check that context was updated in the branch
        assert result.context.scratchpad.get("branch_key") == "yes_branch"
    
    @pytest.mark.fast
    async def test_context_set_with_params(self, tmp_path):
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
        path: "scratchpad.counter"
        value: 42
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")
        
        assert result.success
        assert result.step_history[0].success
        assert result.context.scratchpad.get("counter") == 42
    
    @pytest.mark.fast
    async def test_context_set_with_input(self, tmp_path):
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
      path: "scratchpad.counter"
      value: 99
    updates_context: true
"""
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        runner = create_test_flujo(pipeline)
        result = await gather_result(runner, "test")
        
        assert result.success
        assert result.step_history[0].success
        assert result.context.scratchpad.get("counter") == 99
    
    @pytest.mark.fast
    async def test_statemachine_dynamic_transitions(self, tmp_path):
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
              path: "scratchpad"
              value: { next_state: "middle", step_count: 1 }
            updates_context: true
      
      middle:
        steps:
          - kind: step
            name: goto_final
            agent:
              id: "flujo.builtins.context_merge"
              params:
                path: "scratchpad"
                value: { next_state: "final" }
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
        assert result.context.scratchpad.get("current_state") == "final"
        assert result.context.scratchpad.get("step_count") == 1
        assert result.step_history[-1].name == "test_sm"

