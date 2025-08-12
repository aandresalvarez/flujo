# manual_testing/examples/test_step3.py
"""
Test script for Step 3: Introducing State with PipelineContext

This script tests the stateful clarification loop by running it with
predefined test cases to verify that the state management works correctly
and fixes the limitation from Step 2.
"""

import os
import asyncio
from typing import Optional
from flujo import Flujo
from flujo.domain.models import PipelineResult
from manual_testing.examples.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE, CohortContext


def ensure_api_key():
    """Ensure the API key is loaded from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    print("\u2705 API key configured")


async def test_step3_stateful():
    """Test the Step 3 stateful pipeline with predefined test cases."""

    print("🧪 Testing Step 3: Stateful Pipeline with PipelineContext")
    print("=" * 60)

    # Ensure API key is set
    ensure_api_key()

    # Create the runner with our custom context
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name="test_cohort_clarification_v3",
        context_model=CohortContext,
    )

    # Test cases with different levels of ambiguity
    test_cases = [
        {
            "name": "Clear Definition",
            "definition": "Patients diagnosed with Stage I breast cancer in the last 12 months",
            "expected_clarifications": 0,
        },
        {
            "name": "Moderately Ambiguous",
            "definition": "Patients with asthma, on medication, seen in clinic",
            "expected_clarifications": 2,
        },
        {"name": "Highly Ambiguous", "definition": "cancer patients", "expected_clarifications": 3},
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test Case {i}: {test_case['name']} {'='*20}")
        print(f"Initial Definition: '{test_case['definition']}'")
        print(f"Expected Clarifications: {test_case['expected_clarifications']}")

        # Initialize context with test data
        initial_context_data = {
            "initial_prompt": test_case["definition"],  # Required field from PipelineContext
            "current_definition": test_case["definition"],
            "is_clear": False,
            "clarification_count": 0,
        }

        # Run the pipeline
        result: Optional[PipelineResult] = None
        async for item in runner.run_async(
            test_case["definition"], initial_context_data=initial_context_data
        ):
            result = item

        if result and result.final_pipeline_context:
            final_context = result.final_pipeline_context

            print(f"\n✅ Test Case {i} Results:")
            print(f"  Final Definition: {final_context.current_definition}")
            print(f"  Was Clear: {final_context.is_clear}")
            print(f"  Clarifications Requested: {final_context.clarification_count}")

            # Validate the results
            if final_context.is_clear:
                print("  ✅ SUCCESS: Definition was clarified")
            else:
                print("  ⚠️ WARNING: Definition was not clarified (may have hit max_loops)")

            if final_context.clarification_count >= test_case["expected_clarifications"]:
                print("  ✅ SUCCESS: Expected number of clarifications or more")
            else:
                print("  ⚠️ NOTE: Fewer clarifications than expected")

        else:
            print(f"❌ Test Case {i} FAILED: Pipeline execution failed")

    # Show tracing info for the last test
    if result and result.final_pipeline_context and result.final_pipeline_context.run_id:
        run_id = result.final_pipeline_context.run_id
        print(f"\n📊 Trace saved with ID: {run_id}")
        print(f"To inspect: flujo lens trace {run_id}")


async def test_step3_improvements():
    """Demonstrate the key improvements of Step 3 over Step 2."""

    print("\n" + "=" * 60)
    print("🔍 Step 3 Improvements Over Step 2")
    print("=" * 60)

    print("""
Key Improvements:

1. ✅ STATE MANAGEMENT: The pipeline now has memory
   - Previous: Agent kept asking about the original prompt
   - Now: Agent builds on previous clarifications

2. ✅ AUTONOMOUS EXECUTION: No manual loop simulation needed
   - Previous: Manual loop in main.py to provide clarifications
   - Now: Pipeline handles iteration internally

3. ✅ EXPLICIT STATE: Boolean flags instead of string parsing
   - Previous: Parse '[CLARITY_CONFIRMED]' from agent output
   - Now: Check context.is_clear boolean flag

4. ✅ CONTEXT UPDATES: Steps can modify shared state
   - Previous: Pass data between steps as parameters
   - Now: Steps update shared CohortContext

5. ✅ DATA FLOW CONTROL: Mappers control what data flows where
   - Previous: Simple input/output between steps
   - Now: Explicit mappers control data flow in loops
""")


if __name__ == "__main__":
    asyncio.run(test_step3_stateful())
    asyncio.run(test_step3_improvements())
