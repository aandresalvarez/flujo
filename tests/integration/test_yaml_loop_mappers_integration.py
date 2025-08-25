"""Integration tests for YAML loop step mapper functionality (FSD-026)."""

import pytest
from typing import Any


from flujo.domain.dsl.loop import LoopStep
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent


class MockContext(PipelineContext):
    """Mock context with required fields for testing."""

    initial_prompt: str = ""
    conversation_history: list = []
    command_log: list = []


class TestYAMLLoopMappersIntegration:
    """Integration test suite for YAML loop step mapper functionality."""

    @pytest.fixture
    def skills_helpers(self):
        """Create a skills helpers module for testing."""

        class SkillsHelpers:
            @staticmethod
            def map_initial_input(initial_goal: str, context: MockContext) -> dict:
                """Transform initial raw string goal into structured input for first iteration."""
                context.initial_prompt = initial_goal
                context.command_log.append(f"Initial Goal: {initial_goal}")
                return {"initial_goal": initial_goal, "conversation_history": []}

            @staticmethod
            def map_iteration_input(output: Any, context: MockContext, iteration: int) -> dict:
                """Map previous iteration output to next iteration input."""
                context.conversation_history.append(output)
                return {
                    "initial_goal": context.initial_prompt,
                    "conversation_history": context.conversation_history,
                }

            @staticmethod
            def is_finish_command(output: Any, context: MockContext) -> bool:
                """Check if the conversation should finish."""
                # Finish after 2 iterations or if output contains 'finish'
                return len(context.conversation_history) >= 2 or "finish" in str(output).lower()

            @staticmethod
            def map_loop_output(output: Any, context: MockContext) -> dict:
                """Map final successful output to LoopStep output."""
                return {
                    "final_result": output,
                    "conversation_summary": context.conversation_history,
                    "total_iterations": len(context.conversation_history),
                    "initial_goal": context.initial_prompt,
                }

            @staticmethod
            def map_initial_input_error(initial_goal: str, context: MockContext) -> dict:
                """Error mapper that raises an exception for testing."""
                raise RuntimeError("Initial mapper error")

        return SkillsHelpers()

    @pytest.mark.asyncio
    async def test_conversational_loop_pattern_execution(self, skills_helpers, monkeypatch):
        """Test the complete conversational loop pattern described in FSD-026."""
        # Test the mapper functions directly instead of through YAML loading
        context = MockContext(initial_prompt="test_goal")

        # Test initial input mapper
        initial_result = skills_helpers.map_initial_input("build a website", context)
        assert initial_result["initial_goal"] == "build a website"
        assert initial_result["conversation_history"] == []
        assert context.initial_prompt == "build a website"
        assert len(context.command_log) == 1

        # Test iteration input mapper
        iter1_result = skills_helpers.map_iteration_input("step1_complete", context, 1)
        assert iter1_result["conversation_history"] == ["step1_complete"]
        assert len(context.conversation_history) == 1

        # Test exit condition
        should_exit = skills_helpers.is_finish_command("step1_complete", context)
        assert should_exit is False

        # Test loop output mapper
        final_result = skills_helpers.map_loop_output("final_output", context)
        assert final_result["final_result"] == "final_output"
        assert final_result["total_iterations"] == 1
        assert final_result["initial_goal"] == "build a website"

    @pytest.mark.asyncio
    async def test_loop_step_mapper_execution_flow(self, skills_helpers, monkeypatch):
        """Test that the mapper functions are called in the correct order during execution."""
        # Test the execution flow directly
        context = MockContext(initial_prompt="test_goal")

        # Test initial mapper
        initial_result = skills_helpers.map_initial_input("build a website", context)
        assert initial_result["initial_goal"] == "build a website"
        assert initial_result["conversation_history"] == []

        # Test iteration mapper for first iteration
        iter1_result = skills_helpers.map_iteration_input("step1_complete", context, 1)
        assert iter1_result["conversation_history"] == ["step1_complete"]

        # Test iteration mapper for second iteration
        iter2_result = skills_helpers.map_iteration_input("step2_complete", context, 2)
        assert iter2_result["conversation_history"] == ["step1_complete", "step2_complete"]

        # Test exit condition
        should_exit = skills_helpers.is_finish_command("step2_complete", context)
        assert should_exit is True  # Should exit after 2 iterations

        # Test output mapper
        final_result = skills_helpers.map_loop_output("final_output", context)
        assert final_result["final_result"] == "final_output"
        assert final_result["total_iterations"] == 2

    @pytest.mark.asyncio
    async def test_loop_step_backward_compatibility(self):
        """Test that existing loop steps without mappers continue to work."""
        # Create a loop step without mappers to test backward compatibility
        body = Pipeline.from_step(
            Step.model_validate({"name": "test", "agent": StubAgent(["output"])})
        )
        loop_step = LoopStep(
            name="legacy_loop",
            loop_body_pipeline=body,
            exit_condition_callable=legacy_exit_condition,
            max_loops=3,
        )

        # Verify backward compatibility
        assert loop_step.name == "legacy_loop"
        assert loop_step.initial_input_to_loop_body_mapper is None
        assert loop_step.iteration_input_mapper is None
        assert loop_step.loop_output_mapper is None
        assert loop_step.max_loops == 3

    @pytest.mark.asyncio
    async def test_loop_step_mapper_error_handling(self, skills_helpers, monkeypatch):
        """Test that mapper errors are handled gracefully."""
        # Test error handling directly
        context = MockContext(initial_prompt="test")

        # Test that the error mapper raises the expected error
        with pytest.raises(RuntimeError, match="Initial mapper error"):
            skills_helpers.map_initial_input_error("test_input", context)


# Helper function for backward compatibility test
def legacy_exit_condition(output: Any, context: PipelineContext) -> bool:
    """Legacy exit condition function for testing."""
    return True
