"""
Integration tests for the Cohort Definition Clarification Pipeline.

This test suite verifies that the pipeline correctly handles:
1. Initial assessment of cohort definitions
2. Iterative clarification requests
3. Human-in-the-loop interactions
4. Proper exit conditions
5. Context state management across iterations
"""

import pytest
from unittest.mock import AsyncMock, patch
from flujo import Flujo

# Import the pipeline components
from manual_testing.cohort_pipeline import (
    COHORT_CLARIFICATION_PIPELINE,
    CohortContext,
    HumanClarificationInput,
)


class TestCohortClarificationPipeline:
    """Test suite for the cohort clarification pipeline."""

    @pytest.fixture
    def mock_agent(self):
        """Mock the ClarificationAgent to control responses."""
        with patch("manual_testing.cohort_pipeline.ClarificationAgent") as mock:
            mock.run = AsyncMock()
            yield mock

    @pytest.fixture
    def flujo_runner(self):
        """Create a Flujo runner with the cohort pipeline."""
        return Flujo(COHORT_CLARIFICATION_PIPELINE, context_model=CohortContext)

    @pytest.mark.asyncio
    async def test_clear_definition_first_try(self, mock_agent, flujo_runner):
        """Test that a clear definition exits the loop immediately."""
        # Mock agent to return a clear definition
        mock_agent.run.return_value = "[CLARITY_CONFIRMED] Patients with asthma (ICD-10 J45.x), on medication, seen in clinic between 2020-2023."

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-clear-first-try"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify the pipeline succeeded
        assert result is not None
        assert result.final_pipeline_context is not None

        # Verify context state
        context = result.final_pipeline_context
        assert context.is_clear is True
        assert context.clarification_requests_count == 0
        assert context.status == "clear"
        assert "[CLARITY_CONFIRMED]" not in context.current_definition
        assert "asthma" in context.current_definition.lower()

        # Verify step history
        assert len(result.step_history) > 0
        final_step = result.step_history[-1]
        assert final_step.success is True

    @pytest.mark.asyncio
    async def test_unclear_definition_requires_clarification(self, mock_agent, flujo_runner):
        """Test that an unclear definition triggers clarification request."""
        # Mock agent to return a clarification request
        mock_agent.run.return_value = (
            "Please specify the exact ICD-10 codes for asthma to be included."
        )

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-unclear-definition"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify the pipeline failed due to max_loops (no human input provided)
        assert result is not None
        assert result.final_pipeline_context is not None

        # Verify context state
        context = result.final_pipeline_context
        assert context.is_clear is False
        assert context.clarification_requests_count == 1
        assert context.status == "needs_clarification"
        assert "Please specify" in context.agent_response

        # Verify step history shows failure due to max_loops
        assert len(result.step_history) > 0
        final_step = result.step_history[-1]
        assert final_step.success is False
        assert "max_loops" in final_step.feedback.lower()

    @pytest.mark.asyncio
    async def test_human_in_the_loop_clarification(self, mock_agent, flujo_runner):
        """Test human-in-the-loop clarification flow."""
        # Mock agent to first request clarification, then confirm clarity
        mock_agent.run.side_effect = [
            "Please specify the exact ICD-10 codes for asthma to be included.",
            "[CLARITY_CONFIRMED] Patients with asthma (ICD-10 J45.x), on medication, seen in clinic between 2020-2023.",
        ]

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-human-clarification"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        # Mock human input
        human_input = HumanClarificationInput(
            clarification="Use ICD-10 codes J45.0-J45.9 for asthma diagnosis."
        )

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # The pipeline should pause for human input
        assert result is not None
        assert result.final_pipeline_context is not None

        # Check if pipeline is paused
        if result.final_pipeline_context.scratchpad.get("status") == "paused":
            # Resume with human input
            result = await flujo_runner.resume_async(result, human_input)

        # Verify final state
        assert result is not None
        context = result.final_pipeline_context
        assert context.is_clear is True
        assert context.clarification_requests_count == 1
        assert "asthma" in context.current_definition.lower()
        assert "ICD-10" in context.current_definition

    @pytest.mark.asyncio
    async def test_context_state_management(self, mock_agent, flujo_runner):
        """Test that context state is properly managed across iterations."""
        # Mock agent to request clarification multiple times
        mock_agent.run.return_value = "Please specify the exact time period for the study."

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-context-management"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify context state after max_loops
        assert result is not None
        context = result.final_pipeline_context

        # Verify that context fields are properly initialized
        assert hasattr(context, "current_definition")
        assert hasattr(context, "is_clear")
        assert hasattr(context, "clarification_requests_count")
        assert hasattr(context, "agent_response")
        assert hasattr(context, "status")

        # Verify that the context reflects the final state
        assert context.is_clear is False
        assert context.status == "needs_clarification"
        assert "Please specify" in context.agent_response

    @pytest.mark.asyncio
    async def test_loop_exit_condition_logic(self, mock_agent, flujo_runner):
        """Test that the loop exit condition works correctly."""
        # Mock agent to return clear definition on second call
        mock_agent.run.side_effect = [
            "Please specify the exact ICD-10 codes.",
            "[CLARITY_CONFIRMED] Patients with asthma (ICD-10 J45.x), on medication, seen in clinic.",
        ]

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-exit-condition"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify the pipeline succeeded
        assert result is not None
        context = result.final_pipeline_context
        assert context.is_clear is True
        assert context.clarification_requests_count == 1

        # Verify step history shows success
        final_step = result.step_history[-1]
        assert final_step.success is True

    @pytest.mark.asyncio
    async def test_input_mapping_functions(self, mock_agent, flujo_runner):
        """Test that input mapping functions work correctly."""
        # Mock agent to return clear definition
        mock_agent.run.return_value = "[CLARITY_CONFIRMED] Patients with asthma (ICD-10 J45.x), on medication, seen in clinic."

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-input-mapping"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify that the final output contains the clarified definition
        assert result is not None
        context = result.final_pipeline_context
        assert context.current_definition == result.step_history[-1].output
        assert "asthma" in context.current_definition.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agent, flujo_runner):
        """Test error handling in the pipeline."""
        # Mock agent to raise an exception
        mock_agent.run.side_effect = Exception("Agent failed")

        initial_definition = "Patients with asthma, on medication, seen in clinic."
        run_id = "test-error-handling"
        initial_context_data = {
            "run_id": run_id,
            "initial_prompt": initial_definition,
            "current_definition": initial_definition,
        }

        result = None
        async for item in flujo_runner.run_async(
            initial_definition, run_id=run_id, initial_context_data=initial_context_data
        ):
            result = item

        # Verify that the pipeline handles errors gracefully
        assert result is not None
        final_step = result.step_history[-1]
        assert final_step.success is False
        assert "Agent failed" in final_step.feedback or "Exception" in final_step.feedback
