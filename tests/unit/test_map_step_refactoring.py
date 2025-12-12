"""Tests for the refactored MapStep implementation."""

import pytest
from typing import List
from flujo.domain.dsl import MapStep, Pipeline, Step
from flujo.domain.models import PipelineContext


# Define TestContext outside of test class scope to avoid pytest collection warning
class _TestContext(PipelineContext):
    """Test context with an iterable field."""

    initial_prompt: str = "test prompt"
    items: List[str] = ["item1", "item2", "item3"]


@pytest.fixture
def simple_pipeline() -> Pipeline[str, str]:
    """Create a simple pipeline for testing."""

    async def process_item(item: str) -> str:
        return f"processed_{item}"

    step = Step.from_callable(process_item, name="process")
    return Pipeline.from_step(step)


@pytest.fixture
def map_step(simple_pipeline: Pipeline[str, str]) -> MapStep[_TestContext]:
    """Create a MapStep instance for testing."""
    return MapStep.from_pipeline(name="test_map", pipeline=simple_pipeline, iterable_input="items")


class TestMapStepRefactoring:
    """Test the refactored MapStep implementation."""

    def test_map_step_creation(self, map_step: MapStep[_TestContext]) -> None:
        """Test that MapStep can be created with the new factory method."""
        assert map_step.name == "test_map"
        assert map_step.iterable_input == "items"
        assert map_step.pipeline_to_run is not None

    def test_map_step_properties(self, map_step: MapStep[_TestContext]) -> None:
        """Test that MapStep properties work correctly."""
        # Initially, no items processed
        assert map_step.max_loops == 1
        assert map_step.items is None
        assert map_step.results is None

    def test_map_step_is_complex(self, map_step: MapStep[_TestContext]) -> None:
        """Test that MapStep is marked as complex."""
        assert map_step.is_complex is True

    def test_map_step_repr(self, map_step: MapStep[_TestContext]) -> None:
        """Test MapStep string representation."""
        repr_str = repr(map_step)
        assert "MapStep" in repr_str
        assert "test_map" in repr_str
        assert "items" in repr_str

    def test_map_step_post_init(self, map_step: MapStep[_TestContext]) -> None:
        """Test that post-init sets up internal state correctly."""
        assert map_step.original_body_pipeline is not None
        assert map_step.items is None
        assert map_step.results is None

    def test_map_step_with_empty_iterable(self, simple_pipeline: Pipeline[str, str]) -> None:
        """Test MapStep behavior with empty iterable."""
        _TestContext(initial_prompt="test", items=[])
        map_step = MapStep.from_pipeline(
            name="empty_map", pipeline=simple_pipeline, iterable_input="items"
        )

        # Should create no-op pipeline for empty iterable
        body_pipeline = map_step.loop_body_pipeline
        assert body_pipeline != simple_pipeline  # Should be no-op pipeline

    def test_map_step_mapping_functions(self, map_step: MapStep[_TestContext]) -> None:
        """Test that mapping functions are properly configured."""
        context = _TestContext(initial_prompt="test")

        # Test initial mapper
        initial_mapper = map_step.initial_input_to_loop_body_mapper
        first_item = initial_mapper(None, context)
        assert first_item == "item1"
        assert map_step.items == ["item1", "item2", "item3"]
        assert map_step.results == []
        assert map_step.max_loops == 3

        # Test iteration mapper
        iteration_mapper = map_step.iteration_input_mapper
        second_item = iteration_mapper("result1", context, 1)
        assert second_item == "item2"
        assert map_step.results == ["result1"]

        third_item = iteration_mapper("result2", context, 2)
        assert third_item == "item3"
        assert map_step.results == ["result1", "result2"]

        # Test output mapper
        output_mapper = map_step.loop_output_mapper
        results = output_mapper("result3", context)
        assert results == ["result1", "result2"]

    def test_map_step_error_handling(self, simple_pipeline: Pipeline[str, str]) -> None:
        """Test MapStep error handling."""
        map_step = MapStep.from_pipeline(
            name="error_map", pipeline=simple_pipeline, iterable_input="items"
        )

        # Test with None context
        initial_mapper = map_step.initial_input_to_loop_body_mapper
        with pytest.raises(ValueError, match="MapStep requires a context"):
            initial_mapper(None, None)

        # Test with invalid iterable type
        class InvalidContext(PipelineContext):
            initial_prompt: str = "test"
            items: str = "not_an_iterable"

        invalid_context = InvalidContext(initial_prompt="test")
        with pytest.raises(TypeError, match="must be a non-string iterable"):
            initial_mapper(None, invalid_context)

    def test_map_step_factory_method(self, simple_pipeline: Pipeline[str, str]) -> None:
        """Test the from_pipeline factory method."""
        map_step = MapStep.from_pipeline(
            name="factory_test", pipeline=simple_pipeline, iterable_input="items"
        )

        assert map_step.name == "factory_test"
        assert map_step.pipeline_to_run is not None
        # Compare structure instead of generic-param rendering differences.
        assert list(map_step.pipeline_to_run.steps) == list(simple_pipeline.steps)
        assert map_step.iterable_input == "items"

    def test_map_step_inheritance(self, map_step: MapStep[_TestContext]) -> None:
        """Test that MapStep properly inherits from LoopStep."""
        from flujo.domain.dsl import LoopStep

        assert isinstance(map_step, LoopStep)
        assert hasattr(map_step, "loop_body_pipeline")
        assert hasattr(map_step, "max_loops")
        assert hasattr(map_step, "exit_condition_callable")
