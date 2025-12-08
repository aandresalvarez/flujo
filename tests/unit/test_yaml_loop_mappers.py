"""Tests for YAML loop step mapper functionality (FSD-026)."""

import pytest
from typing import Any, Optional


from flujo.domain.blueprint.loader import (
    load_pipeline_blueprint_from_yaml,
    dump_pipeline_blueprint_to_yaml,
)
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent
from unittest import mock
from flujo.infra.config_manager import FlujoConfig


@pytest.fixture(autouse=True)
def mock_allowed_imports():
    """Allow test modules to be imported during blueprint loading."""
    with mock.patch("flujo.domain.blueprint.loader_resolution.get_config_provider") as mock_get:
        mock_config = mock.Mock(spec=FlujoConfig)
        mock_config.blueprint_allowed_imports = [
            "tests.unit.test_yaml_loop_mappers",
            "invalid.module",
        ]
        mock_config.settings = mock.Mock()
        mock_config.settings.blueprint_allowed_imports = [
            "tests.unit.test_yaml_loop_mappers",
            "invalid.module",
        ]

        mock_get.return_value.load_config.return_value = mock_config
        yield


# Test helper functions for the mappers (not test functions)
def _test_initial_mapper(input_data: Any, context: Optional[PipelineContext]) -> Any:
    """Test initial input mapper function."""
    return "mapped_input_data"


def _test_iteration_mapper(output: Any, context: Optional[PipelineContext], iteration: int) -> Any:
    """Test iteration input mapper function."""
    return f"iteration_{iteration}_{output}"


def _test_output_mapper(output: Any, context: Optional[PipelineContext]) -> Any:
    """Test loop output mapper function."""
    return {"final_output": output, "context": context}


def _test_exit_condition(output: Any, context: Optional[PipelineContext]) -> bool:
    """Test exit condition function."""
    return True


def _map_initial_goal(initial_goal: str, context: PipelineContext) -> dict:
    """Map initial raw string goal to structured input for first iteration."""
    context.initial_prompt = initial_goal
    return {"initial_goal": initial_goal, "conversation_history": []}


def _map_iteration_input(output: Any, context: PipelineContext, iteration: int) -> dict:
    """Map iteration output to next iteration input."""
    if not hasattr(context, "conversation_history"):
        context.conversation_history = []
    context.conversation_history.append(output)
    return {
        "initial_goal": context.initial_prompt,
        "conversation_history": context.conversation_history,
    }


def _is_finish_command(output: Any, context: PipelineContext) -> bool:
    """Check if the conversation should finish."""
    return output == "finish" or (
        hasattr(context, "conversation_history") and len(context.conversation_history) >= 5
    )


def _map_loop_output(output: Any, context: PipelineContext) -> dict:
    """Map final output to loop result."""
    return {
        "final_result": output,
        "conversation_summary": context.conversation_history
        if hasattr(context, "conversation_history")
        else [],
        "total_iterations": len(context.conversation_history)
        if hasattr(context, "conversation_history")
        else 0,
    }


def _validation_mapper(input_data: Any, context: PipelineContext) -> Any:
    """Mapper that marks validation in context for testing."""
    context.validation_called = True
    return f"validated_{input_data}"


class TestYAMLLoopMappers:
    """Test suite for YAML loop step mapper functionality."""

    def test_yaml_loop_with_initial_input_mapper(self) -> None:
        """Test that YAML can define a loop with initial_input_mapper."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: test_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 3
      initial_input_mapper: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        assert len(pipeline.steps) == 1

        loop_step = pipeline.steps[0]
        assert isinstance(loop_step, LoopStep)
        assert loop_step.name == "test_loop"
        assert loop_step.initial_input_to_loop_body_mapper is not None
        assert loop_step.max_loops == 3

    def test_yaml_loop_with_all_mappers(self) -> None:
        """Test that YAML can define a loop with all mapper types."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: comprehensive_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 5
      initial_input_mapper: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      iteration_input_mapper: "tests.unit.test_yaml_loop_mappers:_test_iteration_mapper"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
      loop_output_mapper: "tests.unit.test_yaml_loop_mappers:_test_output_mapper"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        assert len(pipeline.steps) == 1

        loop_step = pipeline.steps[0]
        assert isinstance(loop_step, LoopStep)
        assert loop_step.name == "comprehensive_loop"
        assert loop_step.initial_input_to_loop_body_mapper is not None
        assert loop_step.iteration_input_mapper is not None
        assert loop_step.loop_output_mapper is not None
        assert loop_step.max_loops == 5

    def test_yaml_loop_without_mappers(self) -> None:
        """Test that YAML can define a loop without mappers (backward compatibility)."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: simple_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 2
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        assert len(pipeline.steps) == 1

        loop_step = pipeline.steps[0]
        assert isinstance(loop_step, LoopStep)
        assert loop_step.name == "simple_loop"
        assert loop_step.initial_input_to_loop_body_mapper is None
        assert loop_step.iteration_input_mapper is None
        assert loop_step.loop_output_mapper is None
        assert loop_step.max_loops == 2

    def test_yaml_loop_mapper_import_error(self) -> None:
        """Test that invalid mapper import strings raise appropriate errors."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: invalid_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 1
      initial_input_mapper: "invalid.module:function"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
"""

        with pytest.raises(Exception) as exc_info:
            load_pipeline_blueprint_from_yaml(yaml_text)

        # Should fail during import resolution - check for either error message
        error_msg = str(exc_info.value)
        assert "invalid.module" in error_msg or "No module named 'invalid'" in error_msg

    def test_yaml_loop_mapper_execution(self) -> None:
        """Test that the loaded mappers can be executed correctly."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: executable_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 1
      initial_input_mapper: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        loop_step = pipeline.steps[0]

        # Test that the mapper function works
        context = PipelineContext(initial_prompt="test")
        result = loop_step.initial_input_to_loop_body_mapper("input_data", context)
        assert result == "mapped_input_data"

    def test_yaml_loop_dump_load_roundtrip(self) -> None:
        """Test that loop steps can be dumped and loaded correctly."""
        # Create a loop step with mappers
        body = Pipeline.from_step(
            Step.model_validate({"name": "test", "agent": StubAgent(["output"])})
        )
        loop_step = LoopStep(
            name="test_loop",
            loop_body_pipeline=body,
            exit_condition_callable=_test_exit_condition,
            max_loops=3,
            initial_input_to_loop_body_mapper=_test_initial_mapper,
            iteration_input_mapper=_test_iteration_mapper,
            loop_output_mapper=_test_output_mapper,
        )

        pipeline = Pipeline(steps=[loop_step])

        # Dump to YAML
        yaml_text = dump_pipeline_blueprint_to_yaml(pipeline)

        # Load from YAML
        loaded_pipeline = load_pipeline_blueprint_from_yaml(yaml_text)

        # Verify basic structure is preserved
        assert len(loaded_pipeline.steps) == 1
        loaded_loop = loaded_pipeline.steps[0]
        assert isinstance(loaded_loop, LoopStep)
        assert loaded_loop.name == "test_loop"
        assert loaded_loop.max_loops == 3

    def test_yaml_loop_conversational_pattern(self) -> None:
        """Test the conversational loop pattern described in FSD-026."""
        yaml_text = """
version: "0.1"
steps:
  - kind: hitl
    name: get_initial_goal
    message: "What is your goal?"
  
  - kind: loop
    name: clarification_loop
    loop:
      body:
        - name: planner
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
        - name: executor
          uses: "tests.unit.test_yaml_loop_mappers:_test_iteration_mapper"
      initial_input_mapper: "tests.unit.test_yaml_loop_mappers:_map_initial_goal"
      iteration_input_mapper: "tests.unit.test_yaml_loop_mappers:_map_iteration_input"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_is_finish_command"
      loop_output_mapper: "tests.unit.test_yaml_loop_mappers:_map_loop_output"
      max_loops: 10
  
  - kind: step
    name: generate_specification
    uses: "tests.unit.test_yaml_loop_mappers:_test_output_mapper"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        assert len(pipeline.steps) == 3

        # Verify the loop step has all mappers
        loop_step = pipeline.steps[1]
        assert isinstance(loop_step, LoopStep)
        assert loop_step.name == "clarification_loop"
        assert loop_step.initial_input_to_loop_body_mapper is not None
        assert loop_step.iteration_input_mapper is not None
        assert loop_step.loop_output_mapper is not None
        assert loop_step.max_loops == 10

    def test_yaml_loop_mapper_validation(self) -> None:
        """Test that mapper functions are properly validated and called."""
        yaml_text = """
version: "0.1"
steps:
  - kind: loop
    name: validation_loop
    loop:
      body:
        - kind: step
          name: body_step
          uses: "tests.unit.test_yaml_loop_mappers:_test_initial_mapper"
      max_loops: 1
      initial_input_mapper: "tests.unit.test_yaml_loop_mappers:_validation_mapper"
      exit_condition: "tests.unit.test_yaml_loop_mappers:_test_exit_condition"
"""

        pipeline = load_pipeline_blueprint_from_yaml(yaml_text)
        loop_step = pipeline.steps[0]

        # Test mapper validation - use a custom context that allows dynamic attributes
        class TestContext(PipelineContext):
            validation_called: bool = False

        context = TestContext(initial_prompt="test")

        # Should call the validation mapper
        result = loop_step.initial_input_to_loop_body_mapper("test_input", context)
        assert result == "validated_test_input"
        assert context.validation_called is True
