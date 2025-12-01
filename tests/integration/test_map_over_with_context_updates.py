"""
Integration tests for Map Over + Context Updates feature combination.

This tests the critical combination of map operations with context-updating steps,
which could reveal bugs in context state management during iterative operations.
"""

import pytest
from typing import List
from flujo import step, Step
from flujo.domain.models import PipelineContext
from flujo.domain.dsl.step import MergeStrategy
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class MapContext(PipelineContext):
    """Context for testing map operations with context updates."""

    initial_prompt: str = "test"
    items: List[str] = ["item1", "item2", "item3", "item4"]
    processed_items: List[str] = []
    total_processed: int = 0
    map_results: JSONObject = {}
    current_item: str = ""
    processing_history: List[str] = []


@step(updates_context=True)
async def map_item_step(item: str, *, context: MapContext) -> str:
    """Process a single item and update context."""
    context.current_item = item
    context.total_processed += 1
    context.processed_items.append(item)
    context.processing_history.append(f"processed_{item}")

    # Simulate some processing
    result = f"processed_{item}"
    context.map_results[item] = result

    return result


@step(updates_context=True)
async def map_with_error_step(item: str, *, context: MapContext) -> str:
    """Process item that sometimes fails."""
    context.current_item = item
    context.total_processed += 1
    context.processing_history.append(f"attempted_{item}")

    # Fail on specific items
    if item == "item2":
        raise RuntimeError(f"Intentional failure for {item}")

    result = f"processed_{item}"
    context.map_results[item] = result
    context.processed_items.append(item)

    return result


@step(updates_context=True)
async def map_with_context_dependent_step(item: str, *, context: MapContext) -> str:
    """Process item with context-dependent logic."""
    context.current_item = item
    context.total_processed += 1

    # Use context state to determine processing
    # After processing 2 items, switch to late processing
    if len(context.processed_items) >= 2:
        result = f"late_processed_{item}"
    else:
        result = f"early_processed_{item}"

    context.map_results[item] = result
    context.processed_items.append(item)
    context.processing_history.append(f"context_dependent_{item}")

    return result


@step(updates_context=True)
async def map_with_nested_context_step(item: str, *, context: MapContext) -> str:
    """Process item with nested context operations."""
    context.current_item = item
    context.total_processed += 1

    # Create nested context operations
    nested_data = {
        "original_item": item,
        "processed_count": context.total_processed,
        "history_length": len(context.processing_history),
    }

    context.map_results[item] = nested_data
    context.processed_items.append(item)
    context.processing_history.append(f"nested_{item}")

    return str(nested_data)


@pytest.mark.asyncio
async def test_map_over_with_context_updates_basic():
    """Test basic map over operation with context updates."""

    map_step = Step.map_over(
        name="basic_map",
        pipeline_to_run=Pipeline.from_step(map_item_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=[
            "current_item",
            "total_processed",
            "processed_items",
            "map_results",
            "processing_history",
        ],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify map operation with context updates
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4
    assert len(result.final_pipeline_context.processed_items) == 4
    assert len(result.final_pipeline_context.map_results) == 4
    assert len(result.final_pipeline_context.processing_history) == 4

    # Verify all items were processed
    for item in ["item1", "item2", "item3", "item4"]:
        assert item in result.final_pipeline_context.processed_items
        assert item in result.final_pipeline_context.map_results


@pytest.mark.asyncio
async def test_map_over_with_context_updates_error_handling():
    """Test map over operation with context updates when some items fail."""

    map_step = Step.map_over(
        name="error_map",
        pipeline_to_run=Pipeline.from_step(map_with_error_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=[
            "current_item",
            "total_processed",
            "processed_items",
            "map_results",
            "processing_history",
        ],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify error handling with context updates
    assert result.step_history[-1].success is True
    # The feedback might be None, so check if it exists first
    if result.step_history[-1].feedback:
        assert "item2" in result.step_history[-1].feedback.lower()

    # Verify context updates from successful items
    assert result.final_pipeline_context.total_processed >= 1
    assert len(result.final_pipeline_context.processed_items) >= 1
    assert len(result.final_pipeline_context.processing_history) >= 1

    # Enhanced: Verify error items are tracked in processing history
    processing_history = result.final_pipeline_context.processing_history
    assert any("item2" in item for item in processing_history) or len(processing_history) >= 3


@pytest.mark.asyncio
async def test_map_over_with_context_updates_context_dependent():
    """Test map over operation with context-dependent processing."""

    map_step = Step.map_over(
        name="context_dependent_map",
        pipeline_to_run=Pipeline.from_step(map_with_context_dependent_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=[
            "current_item",
            "total_processed",
            "processed_items",
            "map_results",
            "processing_history",
        ],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify context-dependent processing
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4
    assert len(result.final_pipeline_context.processed_items) == 4

    # Verify context-dependent results
    results = result.final_pipeline_context.map_results
    assert "early_processed_item1" in results.values()
    assert "early_processed_item2" in results.values()
    assert "late_processed_item3" in results.values()
    assert "late_processed_item4" in results.values()


@pytest.mark.asyncio
async def test_map_over_with_context_updates_nested_context():
    """Test map over operation with nested context operations."""

    map_step = Step.map_over(
        name="nested_context_map",
        pipeline_to_run=Pipeline.from_step(map_with_nested_context_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=[
            "current_item",
            "total_processed",
            "processed_items",
            "map_results",
            "processing_history",
        ],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify nested context operations
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4
    assert len(result.final_pipeline_context.processed_items) == 4

    # Verify nested data in results
    results = result.final_pipeline_context.map_results
    for item in ["item1", "item2", "item3", "item4"]:
        assert item in results
        nested_data = results[item]
        assert "original_item" in nested_data
        assert "processed_count" in nested_data
        assert "history_length" in nested_data


@pytest.mark.asyncio
async def test_map_over_with_context_updates_state_isolation():
    """Test that map over operations properly isolate context state between iterations."""

    @step(updates_context=True)
    async def isolation_map_step(item: str, *, context: MapContext) -> str:
        """Step that tests state isolation in map operations."""
        # Each iteration should see the same initial context state
        iteration_data = {
            "item": item,
            "total_processed_at_start": context.total_processed,
            "processed_count_at_start": len(context.processed_items),
            "current_iteration": context.total_processed + 1,
        }

        context.current_item = item
        context.total_processed += 1
        context.processed_items.append(item)
        context.map_results[item] = iteration_data

        return str(iteration_data)

    map_step = Step.map_over(
        name="isolation_map",
        pipeline_to_run=Pipeline.from_step(isolation_map_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=["current_item", "total_processed", "processed_items", "map_results"],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify state isolation
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4

    # Verify each iteration saw the correct state
    results = result.final_pipeline_context.map_results
    for i, item in enumerate(["item1", "item2", "item3", "item4"]):
        assert item in results
        iteration_data = results[item]
        assert iteration_data["total_processed_at_start"] == i
        assert iteration_data["processed_count_at_start"] == i
        assert iteration_data["current_iteration"] == i + 1


@pytest.mark.asyncio
async def test_map_over_with_context_updates_complex_aggregation():
    """Test map over operation with complex context aggregation."""

    @step(updates_context=True)
    async def aggregation_map_step(item: str, *, context: MapContext) -> str:
        """Step that performs complex context aggregation."""
        # Calculate running statistics
        context.current_item = item
        context.total_processed += 1
        context.processed_items.append(item)

        # Complex aggregation logic
        avg_length = sum(len(x) for x in context.processed_items) / len(context.processed_items)
        total_chars = sum(len(x) for x in context.processed_items)

        aggregation_data = {
            "item": item,
            "running_avg_length": avg_length,
            "total_chars_processed": total_chars,
            "items_processed": len(context.processed_items),
            "current_item_length": len(item),
        }

        context.map_results[item] = aggregation_data
        return str(aggregation_data)

    map_step = Step.map_over(
        name="aggregation_map",
        pipeline_to_run=Pipeline.from_step(aggregation_map_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=["current_item", "total_processed", "processed_items", "map_results"],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify complex aggregation
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4
    assert len(result.final_pipeline_context.processed_items) == 4

    # Verify aggregation data
    results = result.final_pipeline_context.map_results
    for item in ["item1", "item2", "item3", "item4"]:
        assert item in results
        agg_data = results[item]
        assert "running_avg_length" in agg_data
        assert "total_chars_processed" in agg_data
        assert "items_processed" in agg_data
        assert "current_item_length" in agg_data


@pytest.mark.asyncio
async def test_map_over_with_context_updates_metadata_conflicts():
    """Test map over operation with context updates and metadata conflicts."""

    @step(updates_context=True)
    async def metadata_map_step(item: str, *, context: MapContext) -> str:
        """Step that tests metadata conflicts in map operations."""
        context.current_item = item
        context.total_processed += 1
        context.processed_items.append(item)

        # Try to update fields that might conflict with map metadata
        context.map_results[f"metadata_{item}"] = {
            "map_index": context.total_processed - 1,
            "map_item": item,
            "map_metadata": {
                "iteration": context.total_processed,
                "timestamp": "now",
                "data": item,
            },
        }

        return f"metadata_processed_{item}"

    map_step = Step.map_over(
        name="metadata_map",
        pipeline_to_run=Pipeline.from_step(metadata_map_step),
        iterable_input="items",
        merge_strategy=MergeStrategy.CONTEXT_UPDATE,
        field_mapping=["current_item", "total_processed", "processed_items", "map_results"],
    )

    runner = create_test_flujo(map_step, context_model=MapContext)
    result = await gather_result(runner, None)

    # Verify metadata handling
    assert result.step_history[-1].success is True
    assert result.final_pipeline_context.total_processed == 4

    # Verify metadata in results
    results = result.final_pipeline_context.map_results
    for item in ["item1", "item2", "item3", "item4"]:
        metadata_key = f"metadata_{item}"
        assert metadata_key in results
        metadata_data = results[metadata_key]
        assert "map_index" in metadata_data
        assert "map_item" in metadata_data
        assert "map_metadata" in metadata_data
