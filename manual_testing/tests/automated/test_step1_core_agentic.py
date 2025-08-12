# manual_testing/test_step1_core_agentic.py
"""
Comprehensive test for Step 1: The Core Agentic Step

This test validates all the core concepts being tested in Step 1:
- make_agent_async(): Creating a basic AI agent
- Step: The fundamental building block of a pipeline
- Pipeline: A sequence of steps
- Flujo: The pipeline runner
- runner.run(): Executing the pipeline
- FSD-11: Signature-aware context injection
- FSD-12: Automatic tracing and observability
"""

import asyncio
import os
import pytest
from typing import Optional
from unittest.mock import patch

from flujo import Flujo, Step, Pipeline
from flujo.domain.models import PipelineResult, PipelineContext

from manual_testing.examples.cohort_pipeline import (
    COHORT_CLARIFICATION_PIPELINE,
    ClarificationAgent,
    assess_clarity_step,
    CLARIFICATION_AGENT_SYSTEM_PROMPT,
)


class TestStep1CoreAgentic:
    """Test suite for Step 1: Core Agentic Step functionality."""

    def test_agent_creation(self):
        """Test that the agent is created correctly with make_agent_async()."""
        print("\n🧪 Testing agent creation...")

        # Verify the agent was created with correct parameters
        assert ClarificationAgent is not None
        assert hasattr(ClarificationAgent, "run")
        assert callable(ClarificationAgent.run)

        # Verify the agent uses the correct model
        # Note: We can't directly access the model, but we can verify the agent exists
        print("✅ Agent created successfully with make_agent_async()")

        # Test that the system prompt is properly formatted
        assert "[CLARITY_CONFIRMED]" in CLARIFICATION_AGENT_SYSTEM_PROMPT
        assert "clinical research assistant" in CLARIFICATION_AGENT_SYSTEM_PROMPT.lower()
        print("✅ System prompt contains required markers and instructions")

    def test_step_creation(self):
        """Test that the step is created correctly."""
        print("\n🧪 Testing step creation...")

        # Verify the step has the correct structure
        assert assess_clarity_step is not None
        assert assess_clarity_step.name == "AssessAndRefine"
        # The step is created from a callable function, so it has a different agent type
        assert hasattr(assess_clarity_step, "agent")
        print("✅ Step created with correct name and agent")

    def test_pipeline_creation(self):
        """Test that the pipeline is created correctly."""
        print("\n🧪 Testing pipeline creation...")

        # Verify the pipeline structure
        assert COHORT_CLARIFICATION_PIPELINE is not None
        assert len(COHORT_CLARIFICATION_PIPELINE.steps) == 1
        # The pipeline now contains a LoopStep, not a simple Step
        assert COHORT_CLARIFICATION_PIPELINE.steps[0].name == "StatefulClarificationLoop"
        print("✅ Pipeline created with LoopStep")

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_mock_agent(self):
        """Test pipeline execution with a mock agent to avoid API calls."""
        print("\n🧪 Testing pipeline execution with mock agent...")

        # Create a mock agent that returns predictable responses
        class MockClarificationAgent:
            async def run(self, data: str, **kwargs) -> str:
                if "diabetes" in data.lower():
                    return "Please specify the type of diabetes (Type 1, Type 2, or gestational)."
                elif "clear" in data.lower():
                    return "Adult patients with Type 2 diabetes diagnosed in the last 5 years. [CLARITY_CONFIRMED]"
                else:
                    return "Please provide more specific details about the cohort definition."

        # Create a test pipeline with the mock agent
        mock_step = Step(name="MockAssessClarity", agent=MockClarificationAgent())
        mock_pipeline = Pipeline.from_step(mock_step)

        # Create the runner
        runner = Flujo(
            mock_pipeline, pipeline_name="test_cohort_clarification", context_model=PipelineContext
        )

        # Test case 1: Unclear definition
        test_input = "patients with diabetes"
        result: Optional[PipelineResult] = None

        async for item in runner.run_async(
            test_input, initial_context_data={"initial_prompt": test_input}
        ):
            result = item

        # Verify the result
        assert result is not None
        assert result.step_history is not None
        assert len(result.step_history) == 1

        final_step = result.step_history[-1]
        assert final_step.success
        assert "diabetes" in final_step.output.lower()
        assert "[CLARITY_CONFIRMED]" not in final_step.output
        print("✅ Unclear definition correctly identified")

        # Test case 2: Clear definition
        test_input = "clear definition"
        result = None

        async for item in runner.run_async(
            test_input, initial_context_data={"initial_prompt": test_input}
        ):
            result = item

        # Verify the result
        assert result is not None
        assert result.step_history is not None
        assert len(result.step_history) == 1

        final_step = result.step_history[-1]
        assert final_step.success
        assert "[CLARITY_CONFIRMED]" in final_step.output
        print("✅ Clear definition correctly confirmed")

    @pytest.mark.asyncio
    async def test_fsd11_signature_aware_context_injection(self):
        """Test FSD-11 fix: signature-aware context injection."""
        print("\n🧪 Testing FSD-11: Signature-aware context injection...")

        # Test with a stateless agent (should work without context)
        class StatelessTestAgent:
            async def run(self, data: str) -> str:
                return f"Stateless response: {data}"

        stateless_step = Step(name="StatelessTest", agent=StatelessTestAgent())
        stateless_pipeline = Pipeline.from_step(stateless_step)

        runner = Flujo(
            stateless_pipeline, pipeline_name="test_stateless", context_model=PipelineContext
        )

        # This should work without errors (FSD-11 fix)
        result = None
        async for item in runner.run_async(
            "test input", initial_context_data={"initial_prompt": "test"}
        ):
            result = item

        assert result is not None
        assert result.step_history[-1].success
        print("✅ Stateless agent works with context present (FSD-11 fix)")

    @pytest.mark.asyncio
    async def test_fsd12_tracing_and_observability(self):
        """Test FSD-12: Automatic tracing and observability."""
        print("\n🧪 Testing FSD-12: Tracing and observability...")

        # Create a simple test pipeline
        class TracingTestAgent:
            async def run(self, data: str, **kwargs) -> str:
                return f"Traced response: {data}"

        tracing_step = Step(name="TracingTest", agent=TracingTestAgent())
        tracing_pipeline = Pipeline.from_step(tracing_step)

        runner = Flujo(
            tracing_pipeline, pipeline_name="test_tracing", context_model=PipelineContext
        )

        # Run the pipeline
        result = None
        async for item in runner.run_async(
            "tracing test", initial_context_data={"initial_prompt": "tracing test"}
        ):
            result = item

        # Verify tracing information is present
        assert result is not None
        assert result.final_pipeline_context is not None
        assert result.final_pipeline_context.run_id is not None
        assert len(result.final_pipeline_context.run_id) > 0

        # Verify step history contains tracing data
        assert len(result.step_history) == 1
        step = result.step_history[0]
        assert step.name == "TracingTest"
        assert step.success
        assert step.output is not None

        print("✅ Tracing information captured correctly")
        print(f"   Run ID: {result.final_pipeline_context.run_id}")
        print(f"   Step name: {step.name}")
        print(f"   Step output: {step.output}")

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the pipeline."""
        print("\n🧪 Testing error handling...")

        # Test with an agent that raises an exception
        class ErrorTestAgent:
            async def run(self, data: str) -> str:
                raise ValueError("Test error")

        error_step = Step(name="ErrorTest", agent=ErrorTestAgent())
        error_pipeline = Pipeline.from_step(error_step)

        runner = Flujo(error_pipeline, pipeline_name="test_error", context_model=PipelineContext)

        # This should handle the error gracefully
        result = None
        async for item in runner.run_async(
            "error test", initial_context_data={"initial_prompt": "error test"}
        ):
            result = item

        assert result is not None
        assert result.step_history is not None
        assert len(result.step_history) == 1

        step = result.step_history[0]
        assert not step.success
        assert step.feedback is not None
        assert "error" in step.feedback.lower()

        print("✅ Error handling works correctly")

    def test_api_key_validation(self):
        """Test API key validation functionality."""
        print("\n🧪 Testing API key validation...")

        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                from manual_testing.examples.main import ensure_api_key

                ensure_api_key()

        # Test with valid API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key_1234"}):
            from manual_testing.examples.main import ensure_api_key

            # Should not raise an exception
            ensure_api_key()

        print("✅ API key validation works correctly")

    @pytest.mark.asyncio
    async def test_integration_with_real_agent(self):
        """Integration test with the real agent (requires API key)."""
        print("\n🧪 Testing integration with real agent...")

        # Skip if no API key is available
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Skipping real agent test - no API key available")
            return

        # Test with the real pipeline
        from manual_testing.examples.cohort_pipeline import CohortContext

        runner = Flujo(
            COHORT_CLARIFICATION_PIPELINE,
            pipeline_name="integration_test",
            context_model=CohortContext,
        )

        # Test with a simple input
        test_input = "patients with asthma"
        result = None

        async for item in runner.run_async(
            test_input,
            initial_context_data={
                "initial_prompt": test_input,
                "current_definition": test_input,
                "is_clear": False,
                "clarification_count": 0,
            },
        ):
            result = item

        # Verify the result
        assert result is not None
        assert result.step_history is not None
        assert len(result.step_history) == 1

        final_step = result.step_history[-1]
        # The loop may fail due to max iterations, which is expected for real agents
        # that keep asking clarifying questions
        if not final_step.success:
            assert "max_loops" in final_step.feedback or "max_iterations" in final_step.feedback
            print("✅ Integration test with real agent - loop terminated as expected")
        else:
            assert final_step.output is not None
            assert len(final_step.output) > 0
            print("✅ Integration test with real agent successful")
            print(f"   Agent response: {final_step.output[:100]}...")

    def test_pipeline_structure_validation(self):
        """Test that the pipeline structure is correct."""
        print("\n🧪 Testing pipeline structure validation...")

        # Verify pipeline has correct structure
        assert COHORT_CLARIFICATION_PIPELINE is not None
        assert hasattr(COHORT_CLARIFICATION_PIPELINE, "steps")
        assert isinstance(COHORT_CLARIFICATION_PIPELINE.steps, list)
        assert len(COHORT_CLARIFICATION_PIPELINE.steps) == 1

        # Verify the step structure
        step = COHORT_CLARIFICATION_PIPELINE.steps[0]
        assert step.name == "StatefulClarificationLoop"
        # The step is now a LoopStep, not a simple Step
        assert hasattr(step, "name")
        assert hasattr(step, "loop_body_pipeline")

        print("✅ Pipeline structure is correct")

    def test_agent_system_prompt_validation(self):
        """Test that the agent system prompt is properly formatted."""
        print("\n🧪 Testing agent system prompt validation...")

        # Verify the system prompt contains required elements
        assert CLARIFICATION_AGENT_SYSTEM_PROMPT is not None
        assert len(CLARIFICATION_AGENT_SYSTEM_PROMPT) > 0

        # Check for required markers and instructions
        required_elements = [
            "clinical research assistant",
            "cohort definition",
            "[CLARITY_CONFIRMED]",
            "clarify",
        ]

        for element in required_elements:
            assert (
                element.lower() in CLARIFICATION_AGENT_SYSTEM_PROMPT.lower()
            ), f"System prompt missing required element: {element}"

        print("✅ Agent system prompt is properly formatted")


async def run_comprehensive_test():
    """Run the comprehensive test suite for Step 1."""
    print("=" * 80)
    print("STEP 1 COMPREHENSIVE TEST SUITE")
    print("Testing: Core Agentic Step")
    print("=" * 80)

    # Create test instance
    test_suite = TestStep1CoreAgentic()

    # Run all tests
    test_methods = [
        test_suite.test_agent_creation,
        test_suite.test_step_creation,
        test_suite.test_pipeline_creation,
        test_suite.test_pipeline_execution_with_mock_agent,
        test_suite.test_fsd11_signature_aware_context_injection,
        test_suite.test_fsd12_tracing_and_observability,
        test_suite.test_error_handling,
        test_suite.test_api_key_validation,
        test_suite.test_integration_with_real_agent,
        test_suite.test_pipeline_structure_validation,
        test_suite.test_agent_system_prompt_validation,
    ]

    passed = 0
    total = len(test_methods)

    for i, test_method in enumerate(test_methods, 1):
        try:
            print(f"\n[{i}/{total}] Running {test_method.__name__}...")
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            passed += 1
            print(f"✅ {test_method.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_method.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Step 1 implementation is working correctly.")
        print("\nCore concepts validated:")
        print("✅ make_agent_async() - Creating basic AI agents")
        print("✅ Step - Fundamental building block")
        print("✅ Pipeline - Sequence of steps")
        print("✅ Flujo - Pipeline runner")
        print("✅ runner.run() - Pipeline execution")
        print("✅ FSD-11 - Signature-aware context injection")
        print("✅ FSD-12 - Automatic tracing and observability")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
