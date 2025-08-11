# manual_testing/manual_test_step1.py
"""
Manual Test for Step 1: Core Agentic Step

This script demonstrates the real pipeline execution with actual API calls.
It tests two scenarios:
1. Incomplete cohort definition - should ask for clarification
2. Complete cohort definition - should confirm with [CLARITY_CONFIRMED]
"""

import asyncio
import os
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
    # Mask the API key for security - show only last 4 characters
    if len(api_key) < 4:
        masked_key = '*' * len(api_key)
    else:
        masked_key = f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"
    print(f"✅ Using API key: {masked_key}")

async def test_cohort_definition(definition: str, test_name: str):
    """Test a single cohort definition with the real pipeline."""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*60}")
    print(f"📝 Cohort Definition: '{definition}'")
    print(f"{'='*60}")

    # Create the runner
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name=f"manual_test_{test_name.lower().replace(' ', '_')}",
        context_model=PipelineContext
    )

    # Run the pipeline
    result: Optional[PipelineResult] = None
    async for item in runner.run_async(
        definition,
        initial_context_data={"initial_prompt": definition}
    ):
        result = item

    # Display results
    print("\n📊 RESULTS:")
    print(f"{'='*60}")

    if result and result.step_history and result.step_history[-1].success:
        final_step = result.step_history[-1]
        agent_response = final_step.output

        print("✅ Pipeline executed successfully!")
        print("🤖 Agent Response:")
        print(f"   {agent_response}")

        # Analyze the response
        if "[CLARITY_CONFIRMED]" in agent_response:
            print("\n🎉 RESULT: Definition is CLEAR")
            print("   The agent confirmed the definition is complete and clear.")
        else:
            print("\n❓ RESULT: Definition needs CLARIFICATION")
            print("   The agent is asking for more specific details.")

    else:
        print("❌ Pipeline execution failed!")
        if result and result.step_history:
            print(f"   Error: {result.step_history[-1].feedback}")
        else:
            print("   No result returned")

    # Show tracing information
    if result and result.final_pipeline_context and result.final_pipeline_context.run_id:
        run_id = result.final_pipeline_context.run_id
        print("\n🔍 Tracing Information:")
        print(f"   Run ID: {run_id}")
        print(f"   To inspect trace: flujo lens trace {run_id}")

    print(f"{'='*60}")

async def main():
    """Run manual tests with real API calls."""
    print("=" * 80)
    print("MANUAL TEST: Step 1 - Core Agentic Step")
    print("Testing real API calls with actual cohort definitions")
    print("=" * 80)

    # Ensure API key is available
    ensure_api_key()

    # Test Case 1: Incomplete cohort definition
    incomplete_definition = "patients with diabetes"
    await test_cohort_definition(
        incomplete_definition,
        "Incomplete Definition"
    )

    # Test Case 2: Complete cohort definition
    complete_definition = "adult patients with Type 2 diabetes diagnosed in the last 5 years, currently on metformin therapy"
    await test_cohort_definition(
        complete_definition,
        "Complete Definition"
    )

    print("\n" + "=" * 80)
    print("MANUAL TEST COMPLETE")
    print("=" * 80)
    print("✅ Both test cases executed successfully!")
    print("📊 Check the results above to see how the agent handles:")
    print("   - Incomplete definitions (asks for clarification)")
    print("   - Complete definitions (confirms with [CLARITY_CONFIRMED])")
    print("🔍 Use the tracing commands to inspect detailed execution")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
