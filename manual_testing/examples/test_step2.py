# manual_testing/examples/test_step2.py
"""
Test script for Step 2: Adding the Clarification Loop

This script tests the iterative clarification loop by running it with
a predefined test case to verify the loop functionality works correctly.
"""

import os
import asyncio
from typing import Optional
from flujo import Flujo
from flujo.domain.models import PipelineResult, PipelineContext
from manual_testing.examples.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE


def ensure_api_key():
    """Ensure the API key is loaded from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    print("\u2705 API key configured")


async def test_step2_loop():
    """Test the Step 2 clarification loop with a predefined test case."""

    print("🧪 Testing Step 2: Clarification Loop")
    print("=" * 50)

    # Ensure API key is set
    ensure_api_key()

    # Create the runner
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name="test_cohort_clarification_v2",
        context_model=PipelineContext,
    )

    # Test case: Start with an ambiguous definition
    test_definition = "cancer patients"
    print(f"Test Definition: '{test_definition}'")
    print("\nExpected behavior:")
    print("- Agent should ask for clarification about cancer type")
    print("- Loop should continue until definition is clear")
    print("- Should reach max_loops (5) or find clarity")

    # Simulate the interactive loop
    current_definition = test_definition
    final_output = ""

    for i in range(5):  # Limit interactions for this demo
        print(f"\n--- Iteration {i+1} ---")

        # Run the pipeline
        result: Optional[PipelineResult] = None
        async for item in runner.run_async(
            current_definition, initial_context_data={"initial_prompt": current_definition}
        ):
            result = item

        if result and result.step_history and result.step_history[-1].success:
            agent_response = result.step_history[-1].output
            print(f"Agent Response: {agent_response}")

            # Check if the agent has confirmed the definition is clear
            if "[CLARITY_CONFIRMED]" in agent_response:
                print("✅ Agent has confirmed the definition is clear.")
                final_output = agent_response
                break
            else:
                # Simulate user providing clarification
                if i == 0:
                    clarification = "breast cancer"
                elif i == 1:
                    clarification = "stage I and II"
                else:
                    clarification = "diagnosed in the last 2 years"

                print(f"Simulated User Clarification: {clarification}")
                current_definition = f"Previous definition: '{current_definition}'. My clarification is: '{clarification}'"
        else:
            print("❌ Pipeline execution failed.")
            if result and result.step_history:
                print(f"Feedback: {result.step_history[-1].feedback}")
            break
    else:
        print("⚠️ Reached max interactions without clarity.")
        final_output = "Loop finished without confirmation."

    print("\n--- Test Result ---")
    if final_output and "[CLARITY_CONFIRMED]" in final_output:
        final_definition = final_output.replace("[CLARITY_CONFIRMED]", "").strip()
        print("✅ SUCCESS: Final definition is clear")
        print(f"Final Definition: {final_definition}")
    else:
        print("❌ FAILED: Definition was not clarified")

    # Show tracing info
    if result and result.final_pipeline_context and result.final_pipeline_context.run_id:
        run_id = result.final_pipeline_context.run_id
        print(f"\n📊 Trace saved with ID: {run_id}")
        print(f"To inspect: flujo lens trace {run_id}")


if __name__ == "__main__":
    asyncio.run(test_step2_loop())
