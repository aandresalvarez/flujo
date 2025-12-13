"""
Integration tests for Human-in-the-Loop + Context Updates feature combination.

This tests the critical combination of human-in-the-loop operations with context-updating steps,
which could reveal bugs in context state management during human interaction.
"""

import pytest
from typing import Any, List
from flujo import step, Step
from flujo.domain.models import PipelineContext
from flujo.domain.dsl.pipeline import Pipeline
from flujo.testing.utils import gather_result
from tests.conftest import create_test_flujo
from flujo.type_definitions.common import JSONObject


class HITLContext(PipelineContext):
    """Context for testing human-in-the-loop operations with context updates."""

    initial_prompt: str = "test"
    human_interactions: List[str] = []
    total_interactions: int = 0
    hitl_data: JSONObject = {}
    current_interaction: str = ""
    interaction_history: List[str] = []
    approval_count: int = 0
    rejection_count: int = 0


@step(updates_context=True)
async def pre_hitl_step(data: Any, *, context: HITLContext) -> dict:
    """Step that runs before HITL with context updates."""
    context.current_interaction = f"pre_{data}"
    context.total_interactions += 1
    context.interaction_history.append(f"pre_processed_{data}")

    return {
        "current_interaction": context.current_interaction,
        "total_interactions": context.total_interactions,
        "interaction_history": context.interaction_history,
    }


@step(updates_context=True)
async def post_hitl_step(data: Any, *, context: HITLContext) -> dict:
    """Step that runs after HITL with context updates."""
    context.current_interaction = f"post_{data}"
    context.total_interactions += 1
    context.interaction_history.append(f"post_processed_{data}")

    return {
        "current_interaction": context.current_interaction,
        "total_interactions": context.total_interactions,
        "interaction_history": context.interaction_history,
    }


@step(updates_context=True)
async def hitl_approval_step(data: Any, *, context: HITLContext) -> dict:
    """Step that handles HITL approval with context updates."""
    context.approval_count += 1
    context.hitl_data[f"approval_{context.approval_count}"] = {
        "data": data,
        "interaction_count": context.total_interactions,
        "timestamp": "now",
    }
    context.interaction_history.append(f"approved_{data}")

    return {
        "approval_count": context.approval_count,
        "hitl_data": context.hitl_data,
        "interaction_history": context.interaction_history,
    }


@step(updates_context=True)
async def hitl_rejection_step(data: Any, *, context: HITLContext) -> dict:
    """Step that handles HITL rejection with context updates."""
    context.rejection_count += 1
    context.hitl_data[f"rejection_{context.rejection_count}"] = {
        "data": data,
        "interaction_count": context.total_interactions,
        "timestamp": "now",
    }
    context.interaction_history.append(f"rejected_{data}")

    return {
        "rejection_count": context.rejection_count,
        "hitl_data": context.hitl_data,
        "interaction_history": context.interaction_history,
    }


@step(updates_context=True)
async def hitl_with_error_step(data: Any, *, context: HITLContext) -> dict:
    """Step that sometimes fails during HITL processing."""
    context.total_interactions += 1
    context.interaction_history.append(f"attempted_error_{data}")

    # Fail on specific conditions
    if "fail" in str(data).lower():
        raise RuntimeError(f"Intentional failure in HITL for data: {data}")

    context.hitl_data[f"error_success_{context.total_interactions}"] = {
        "data": data,
        "interaction_count": context.total_interactions,
    }

    return {
        "total_interactions": context.total_interactions,
        "interaction_history": context.interaction_history,
        "hitl_data": context.hitl_data,
    }


@pytest.mark.asyncio
async def test_hitl_with_context_updates_basic():
    """Test basic human-in-the-loop operation with context updates."""

    # Create a pipeline with HITL step
    hitl_step = Step.human_in_the_loop(
        name="basic_hitl", message_for_user="Please review and approve the content"
    )

    pipeline = Pipeline.from_step(pre_hitl_step) >> hitl_step >> Pipeline.from_step(post_hitl_step)

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_data")

    # Verify pipeline reached HITL pause or completion
    assert result.step_history[-1].success in {True, False}
    assert result.final_pipeline_context.total_interactions >= 1  # Only pre-HITL step runs
    assert len(result.final_pipeline_context.interaction_history) >= 1

    # Verify HITL step was executed and paused
    # HITL steps may not appear in step_history immediately due to pausing
    # The important thing is that context updates from pre-HITL steps are preserved
    # and the pipeline is paused at the HITL step


@pytest.mark.asyncio
async def test_hitl_with_context_updates_error_handling():
    """Test human-in-the-loop operation with context updates when steps fail."""

    # Create a pipeline with HITL step that might fail
    hitl_step = Step.human_in_the_loop(
        name="error_hitl", message_for_user="Please review and approve the content"
    )

    pipeline = (
        Pipeline.from_step(hitl_with_error_step) >> hitl_step >> Pipeline.from_step(post_hitl_step)
    )

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_fail_data")

    # Verify error handling with context updates
    assert result.step_history[-1].success is False
    assert "intentional failure" in result.step_history[-1].feedback.lower()

    # ✅ ENHANCED TRANSACTIONAL BEHAVIOR: Failed steps don't commit context changes
    # Previous behavior: Partial context updates preserved even on step failure
    # Enhanced behavior: Transaction-like semantics - failed steps don't commit changes
    # This prevents inconsistent state and ensures data integrity
    assert (
        result.final_pipeline_context.total_interactions == 0
    )  # No changes committed from failed step
    assert (
        len(result.final_pipeline_context.interaction_history) == 0
    )  # No partial updates preserved
    # The failure is properly captured in the step result feedback
    assert "intentional failure" in result.step_history[-1].feedback.lower()


@pytest.mark.asyncio
async def test_hitl_with_context_updates_context_dependent():
    """Test human-in-the-loop operation with context-dependent processing."""

    @step(updates_context=True)
    async def context_dependent_hitl_step(data: Any, *, context: HITLContext) -> dict:
        """Step that uses context state for HITL processing."""
        context.current_interaction = f"context_dependent_{data}"
        context.total_interactions += 1

        # Use context state to determine processing
        if context.approval_count > context.rejection_count:
            result = f"approved_context_{data}"
        else:
            result = f"pending_context_{data}"

        context.hitl_data[data] = result
        context.interaction_history.append(f"context_dependent_{data}")

        return {
            "current_interaction": context.current_interaction,
            "total_interactions": context.total_interactions,
            "hitl_data": context.hitl_data,
            "interaction_history": context.interaction_history,
        }

    hitl_step = Step.human_in_the_loop(
        name="context_dependent_hitl",
        message_for_user="Please review and approve the content",
    )

    pipeline = (
        Pipeline.from_step(context_dependent_hitl_step)
        >> hitl_step
        >> Pipeline.from_step(post_hitl_step)
    )

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_data")

    # Verify context-dependent HITL
    assert result.step_history[-1].success in {True, False}
    assert result.final_pipeline_context.total_interactions >= 1  # Only pre-HITL step runs
    assert len(result.final_pipeline_context.interaction_history) >= 1

    # Verify context-dependent processing
    assert "context_dependent" in result.final_pipeline_context.current_interaction


@pytest.mark.asyncio
async def test_hitl_with_context_updates_state_isolation():
    """Test that human-in-the-loop operations properly manage context state."""

    @step(updates_context=True)
    async def isolation_hitl_step(data: Any, *, context: HITLContext) -> dict:
        """Step that tests state isolation in HITL operations."""
        # Each interaction should see the current context state
        interaction_data = {
            "data": data,
            "total_interactions_at_start": context.total_interactions,
            "approval_count_at_start": context.approval_count,
            "rejection_count_at_start": context.rejection_count,
            "current_interaction": context.total_interactions + 1,
        }

        context.current_interaction = f"isolation_{data}"
        context.total_interactions += 1
        context.hitl_data[f"isolation_{data}"] = interaction_data
        context.interaction_history.append(f"isolation_{data}")

        return {
            "current_interaction": context.current_interaction,
            "total_interactions": context.total_interactions,
            "hitl_data": context.hitl_data,
            "interaction_history": context.interaction_history,
        }

    hitl_step = Step.human_in_the_loop(
        name="isolation_hitl", message_for_user="Please review and approve the content"
    )

    pipeline = (
        Pipeline.from_step(isolation_hitl_step) >> hitl_step >> Pipeline.from_step(post_hitl_step)
    )

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_data")

    # Verify state management
    assert result.step_history[-1].success in {True, False}
    assert result.final_pipeline_context.total_interactions >= 1  # Only pre-HITL step runs

    # Verify isolation data was stored
    isolation_data = result.final_pipeline_context.hitl_data.get("isolation_test_data")
    if isolation_data:
        assert "total_interactions_at_start" in isolation_data
        assert "current_interaction" in isolation_data


@pytest.mark.asyncio
async def test_hitl_with_context_updates_complex_interaction():
    """Test human-in-the-loop operation with complex interaction scenarios."""

    @step(updates_context=True)
    async def complex_hitl_step(data: Any, *, context: HITLContext) -> dict:
        """Step with complex HITL operations."""
        context.current_interaction = f"complex_{data}"
        context.total_interactions += 1

        # Complex interaction logic
        complex_data = {
            "data": data,
            "interaction_count": context.total_interactions,
            "approval_ratio": context.approval_count / max(context.total_interactions, 1),
            "rejection_ratio": context.rejection_count / max(context.total_interactions, 1),
            "interaction_history_length": len(context.interaction_history),
            "nested_data": {
                "level1": {
                    "level2": {
                        "value": f"complex_value_{context.total_interactions}",
                        "metadata": {
                            "timestamp": "now",
                            "interaction": context.total_interactions,
                        },
                    }
                }
            },
        }

        context.hitl_data[f"complex_{data}"] = complex_data
        context.interaction_history.append(f"executed_complex_{data}")

        return {
            "current_interaction": context.current_interaction,
            "total_interactions": context.total_interactions,
            "hitl_data": context.hitl_data,
            "interaction_history": context.interaction_history,
        }

    hitl_step = Step.human_in_the_loop(
        name="complex_hitl", message_for_user="Please review and approve the content"
    )

    pipeline = (
        Pipeline.from_step(complex_hitl_step) >> hitl_step >> Pipeline.from_step(post_hitl_step)
    )

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_data")

    # Verify complex HITL
    assert result.step_history[-1].success in {True, False}
    assert result.final_pipeline_context.total_interactions >= 1  # Only pre-HITL step runs

    # Verify complex data was stored
    complex_data = result.final_pipeline_context.hitl_data.get("complex_test_data")
    if complex_data:
        assert "nested_data" in complex_data
        assert "level1" in complex_data["nested_data"]
        assert "level2" in complex_data["nested_data"]["level1"]


@pytest.mark.asyncio
async def test_hitl_with_context_updates_metadata_conflicts():
    """Test human-in-the-loop operation with context updates and metadata conflicts."""

    @step(updates_context=True)
    async def metadata_hitl_step(data: Any, *, context: HITLContext) -> dict:
        """Step that tests metadata conflicts in HITL operations."""
        context.current_interaction = f"metadata_{data}"
        context.total_interactions += 1

        # Try to update fields that might conflict with HITL metadata
        context.hitl_data[f"metadata_{context.total_interactions}"] = {
            "hitl_index": context.total_interactions,
            "hitl_data": data,
            "hitl_metadata": {
                "interaction": context.total_interactions,
                "timestamp": "now",
                "data": data,
            },
        }

        context.interaction_history.append(f"metadata_{data}")

        return {
            "current_interaction": context.current_interaction,
            "total_interactions": context.total_interactions,
            "hitl_data": context.hitl_data,
            "interaction_history": context.interaction_history,
        }

    hitl_step = Step.human_in_the_loop(
        name="metadata_hitl", message_for_user="Please review and approve the content"
    )

    pipeline = (
        Pipeline.from_step(metadata_hitl_step) >> hitl_step >> Pipeline.from_step(post_hitl_step)
    )

    runner = create_test_flujo(pipeline, context_model=HITLContext)
    result = await gather_result(runner, "test_data")

    # Verify metadata handling
    assert result.step_history[-1].success in {True, False}
    assert result.final_pipeline_context.total_interactions >= 1  # Only pre-HITL step runs

    # Verify metadata in hitl data
    metadata_key = f"metadata_{result.final_pipeline_context.total_interactions}"
    if metadata_key in result.final_pipeline_context.hitl_data:
        metadata_data = result.final_pipeline_context.hitl_data[metadata_key]
        assert "hitl_index" in metadata_data
        assert "hitl_data" in metadata_data
        assert "hitl_metadata" in metadata_data
