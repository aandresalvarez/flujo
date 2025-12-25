"""Tests for blueprint loader validation (Task 1.2: Async Function Validation)"""

import pytest
from flujo.domain.blueprint.loader import load_pipeline_blueprint_from_yaml, BlueprintError
from unittest import mock
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        # Allow importing from tests.unit.test_blueprint_loader
        mock_config.blueprint_allowed_imports = ["tests.unit.test_blueprint_loader"]
        # Also need settings object if accessed
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = ["tests.unit.test_blueprint_loader"]

        mock_get.return_value.load_config.return_value = mock_config
        yield


# Test fixtures: async and sync functions for testing
async def async_exit_condition(output, context):
    """Invalid: Async exit condition function"""
    return True


def sync_exit_condition(output, context):
    """Valid: Sync exit condition function"""
    return True


async def async_condition_function(data, context):
    """Invalid: Async condition function"""
    return "branch_a"


def sync_condition_function(data, context):
    """Valid: Sync condition function"""
    return "branch_a"


def tree_search_cost_fn(candidate, parent, depth, evaluation):
    return float(depth)


def tree_search_validator(candidate):
    _ = candidate
    return True


def tree_search_proposer(data):
    _ = data
    return ["a", "b"]


def tree_search_evaluator(data):
    _ = data
    return 1.0


class TestAsyncExitConditionRejection:
    """Test that async exit_condition functions are rejected at load time."""

    def test_async_exit_condition_raises_blueprint_error(self):
        """Test that async function in exit_condition raises BlueprintError with helpful message."""
        yaml_content = """
version: "0.1"
name: "test_async_exit_loop"
steps:
  - kind: loop
    name: "test_loop"
    loop:
      max_loops: 3
      exit_condition: "tests.unit.test_blueprint_loader:async_exit_condition"
      body:
        - kind: step
          name: "dummy_step"
          agent: "flujo.builtins.passthrough"
"""
        with pytest.raises(BlueprintError) as exc_info:
            load_pipeline_blueprint_from_yaml(yaml_content)

        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "must be synchronous" in error_msg
        assert "async def" in error_msg
        assert "def my_condition" in error_msg
        assert "Remove 'async'" in error_msg
        assert "loops#exit-conditions" in error_msg

    def test_sync_exit_condition_accepted(self):
        """Test that sync function in exit_condition is accepted."""
        yaml_content = """
version: "0.1"
name: "test_sync_exit_loop"
steps:
  - kind: loop
    name: "test_loop"
    loop:
      max_loops: 3
      exit_condition: "tests.unit.test_blueprint_loader:sync_exit_condition"
      body:
        - kind: step
          name: "dummy_step"
          agent: "flujo.builtins.passthrough"
"""
        # Should not raise
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        assert pipeline is not None
        assert pipeline.name == "test_sync_exit_loop"

    def test_exit_expression_still_works(self):
        """Test that exit_expression (non-callable) still works correctly."""
        yaml_content = """
version: "0.1"
name: "test_exit_expression"
steps:
  - kind: loop
    name: "test_loop"
    loop:
      max_loops: 3
      exit_expression: "context.counter >= 3"
      body:
        - kind: step
          name: "dummy_step"
          agent: "flujo.builtins.passthrough"
"""
        # Should not raise
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        assert pipeline is not None


class TestAsyncConditionRejection:
    """Test that async condition functions in conditional steps are rejected at load time."""

    def test_async_condition_raises_blueprint_error(self):
        """Test that async function in condition raises BlueprintError with helpful message."""
        yaml_content = """
version: "0.1"
name: "test_async_condition"
steps:
  - kind: conditional
    name: "test_conditional"
    condition: "tests.unit.test_blueprint_loader:async_condition_function"
    branches:
      branch_a:
        - kind: step
          name: "step_a"
          agent: "flujo.builtins.passthrough"
      branch_b:
        - kind: step
          name: "step_b"
          agent: "flujo.builtins.passthrough"
"""
        with pytest.raises(BlueprintError) as exc_info:
            load_pipeline_blueprint_from_yaml(yaml_content)

        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "must be synchronous" in error_msg
        assert "async def" in error_msg
        assert "def my_condition" in error_msg
        assert "Remove 'async'" in error_msg
        assert "pipeline_branching#conditional-steps" in error_msg

    def test_sync_condition_accepted(self):
        """Test that sync function in condition is accepted."""
        yaml_content = """
version: "0.1"
name: "test_sync_condition"
steps:
  - kind: conditional
    name: "test_conditional"
    condition: "tests.unit.test_blueprint_loader:sync_condition_function"
    branches:
      branch_a:
        - kind: step
          name: "step_a"
          agent: "flujo.builtins.passthrough"
      branch_b:
        - kind: step
          name: "step_b"
          agent: "flujo.builtins.passthrough"
"""
        # Should not raise
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        assert pipeline is not None
        assert pipeline.name == "test_sync_condition"

    def test_condition_expression_still_works(self):
        """Test that condition_expression (non-callable) still works correctly."""
        yaml_content = """
version: "0.1"
name: "test_condition_expression"
steps:
  - kind: conditional
    name: "test_conditional"
    condition_expression: "context.value == 'a'"
    branches:
      "true":
        - kind: step
          name: "step_true"
          agent: "flujo.builtins.passthrough"
      "false":
        - kind: step
          name: "step_false"
          agent: "flujo.builtins.passthrough"
"""
        # Should not raise
        pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
        assert pipeline is not None


class TestErrorMessageQuality:
    """Test that error messages are clear and actionable."""


def test_tree_search_blueprint_loads() -> None:
    yaml_content = """
version: "0.1"
name: "tree_search_pipeline"
steps:
  - kind: tree_search
    name: "tree_search"
    proposer: "tests.unit.test_blueprint_loader:tree_search_proposer"
    evaluator: "tests.unit.test_blueprint_loader:tree_search_evaluator"
    cost_function: "tests.unit.test_blueprint_loader:tree_search_cost_fn"
    candidate_validator: "tests.unit.test_blueprint_loader:tree_search_validator"
    branching_factor: 2
    beam_width: 1
    max_depth: 1
    max_iterations: 3
    path_max_tokens: 300
    goal_score_threshold: 0.9
    require_goal: false
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    assert pipeline is not None
    from flujo.domain.dsl.tree_search import TreeSearchStep

    assert isinstance(pipeline.steps[0], TreeSearchStep)


def test_parallel_reduce_majority_vote_loads() -> None:
    yaml_content = """
version: "0.1"
name: "parallel_reduce_pipeline"
steps:
  - kind: parallel
    name: "parallel_step"
    reduce: "majority_vote"
    branches:
      a:
        - kind: step
          name: "step_a"
          agent: "flujo.builtins.passthrough"
      b:
        - kind: step
          name: "step_b"
          agent: "flujo.builtins.passthrough"
"""
    pipeline = load_pipeline_blueprint_from_yaml(yaml_content)
    from flujo.domain.dsl.parallel import ParallelStep

    assert isinstance(pipeline.steps[0], ParallelStep)
    assert callable(pipeline.steps[0].reduce)

    def test_exit_condition_error_includes_helpful_example(self):
        """Test that error message includes code example and documentation link."""
        yaml_content = """
version: "0.1"
name: "test_error_message"
steps:
  - kind: loop
    name: "test_loop"
    loop:
      exit_condition: "tests.unit.test_blueprint_loader:async_exit_condition"
      body:
        - kind: step
          name: "dummy"
          agent: "flujo.builtins.passthrough"
"""
        with pytest.raises(BlueprintError) as exc_info:
            load_pipeline_blueprint_from_yaml(yaml_content)

        error_msg = str(exc_info.value)

        # Check for all important elements
        assert "async_exit_condition" in error_msg  # Function name
        assert "Change your function from:" in error_msg  # Helpful guidance
        assert "async def my_condition" in error_msg  # Bad example
        assert "To:" in error_msg  # Transition
        assert "def my_condition" in error_msg  # Good example
        assert "https://flujo.dev" in error_msg  # Documentation link

    def test_condition_error_includes_helpful_example(self):
        """Test that condition error message includes code example and documentation link."""
        yaml_content = """
version: "0.1"
name: "test_error_message"
steps:
  - kind: conditional
    name: "test_conditional"
    condition: "tests.unit.test_blueprint_loader:async_condition_function"
    branches:
      branch_a:
        - kind: step
          name: "step_a"
          agent: "flujo.builtins.passthrough"
"""
        with pytest.raises(BlueprintError) as exc_info:
            load_pipeline_blueprint_from_yaml(yaml_content)

        error_msg = str(exc_info.value)

        # Check for all important elements
        assert "async_condition_function" in error_msg  # Function name
        assert "Change your function from:" in error_msg  # Helpful guidance
        assert "async def my_condition" in error_msg  # Bad example
        assert "To:" in error_msg  # Transition
        assert "def my_condition" in error_msg  # Good example
        assert "https://flujo.dev" in error_msg  # Documentation link
