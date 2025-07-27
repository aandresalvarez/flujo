# manual_testing/main.py
import os
import asyncio
from typing import Optional
from flujo import Flujo
from flujo.domain.models import PipelineResult
from manual_testing.examples.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE, CohortContext

# Ensure the API key is loaded from environment
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
    print(f"\u2705 Using API key: {masked_key}")

async def main():
    # Ensure the correct API key is set
    ensure_api_key()

    print("Step 3: Running a stateful loop with PipelineContext.\n" + "="*50)

    # --- STEP 3: STATE MANAGEMENT ---
    # 
    # The pipeline now manages state internally using CohortContext.
    # This is much simpler than Step 2 because we don't need to manually
    # simulate the interaction - the pipeline itself manages the evolving state.
    # 
    # Key changes:
    # - Tell the runner about our custom context model
    # - Provide initial context data
    # - The pipeline runs autonomously, refining the definition based on its own feedback

    # Tell the runner about our context model
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name="cohort_clarification_v3",
        context_model=CohortContext  # NEW: Specify our custom context
    )

    # Provide a test definition that will need clarification
    initial_definition = "Patients with asthma, on medication, seen in clinic."

    # Provide the initial data for our context
    # This is how we initialize the state that the pipeline will manage
    initial_context_data = {
        "initial_prompt": initial_definition,  # Required field from PipelineContext
        "current_definition": initial_definition,
        "is_clear": False,
        "clarification_count": 0
    }

    print(f"Initial Definition: {initial_definition}")
    print("\nThe pipeline will now run autonomously, refining the definition...")
    
    # Run the pipeline - it will handle the iteration internally
    result: Optional[PipelineResult] = None
    async for item in runner.run_async(
        initial_definition,
        initial_context_data=initial_context_data
    ):
        result = item

    print("\n--- Final Result ---")
    if result and result.final_pipeline_context:
        final_context = result.final_pipeline_context
        print(f"Final Definition:\n{final_context.current_definition}")
        print(f"Was Clear: {final_context.is_clear}")
        print(f"Clarifications Requested: {final_context.clarification_count}")
    else:
        print("Pipeline execution failed.")

    # --- FSD-12: TRACING DEMONSTRATION ---
    print("\n" + "="*50)
    print("✨ OBSERVABILITY (FSD-12) ✨")
    if result and result.final_pipeline_context and result.final_pipeline_context.run_id:
        run_id = result.final_pipeline_context.run_id
        print(f"Pipeline run completed with ID: {run_id}")
        print("A trace has been saved to the local `flujo_ops.db` file.")
        print("\nTo inspect the trace, run this command in your terminal:")
        print(f"\n  flujo lens trace {run_id}\n")
    else:
        print("Could not retrieve run_id for tracing.")


if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY (or other) set in your .env file
    asyncio.run(main())
