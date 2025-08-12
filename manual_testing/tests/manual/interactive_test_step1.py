# manual_testing/interactive_test_step1.py
"""
Interactive Manual Test for Step 1: Core Agentic Step

This script allows you to input your own cohort definitions and see
how the agent responds in real-time.
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
        masked_key = "*" * len(api_key)
    else:
        masked_key = f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"
    print(f"✅ Using API key: {masked_key}")


async def test_cohort_definition(definition: str):
    """Test a single cohort definition with the real pipeline."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: '{definition}'")
    print(f"{'='*60}")

    # Create the runner
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name="interactive_test",
        context_model=PipelineContext,
    )

    # Run the pipeline
    result: Optional[PipelineResult] = None
    async for item in runner.run_async(
        definition, initial_context_data={"initial_prompt": definition}
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
    """Run interactive manual tests with real API calls."""
    print("=" * 80)
    print("INTERACTIVE MANUAL TEST: Step 1 - Core Agentic Step")
    print("Enter your own cohort definitions to test the pipeline")
    print("=" * 80)

    # Ensure API key is available
    ensure_api_key()

    print("\n💡 Example cohort definitions to try:")
    print("   - 'sick people' (very vague)")
    print("   - 'cancer patients' (incomplete)")
    print("   - 'patients with heart problems' (ambiguous)")
    print(
        "   - 'adult patients aged 18-65 with confirmed Type 2 diabetes diagnosed between 2020-2024, currently prescribed metformin at a dose of 500-2000mg daily, with HbA1c levels between 7.0-10.0%' (complete)"
    )
    print("\n" + "=" * 80)

    while True:
        print(f"\n{'='*60}")
        print("Enter a cohort definition (or 'quit' to exit):")
        print(f"{'='*60}")

        definition = input("📝 Cohort Definition: ").strip()

        if definition.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not definition:
            print("❌ Please enter a valid cohort definition.")
            continue

        # Test the definition
        await test_cohort_definition(definition)

        # Ask if user wants to continue
        print(f"\n{'='*60}")
        continue_test = input("Test another definition? (y/n): ").strip().lower()
        if continue_test not in ["y", "yes", ""]:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
