# manual_testing/main.py
import os
import asyncio
from typing import Optional
from flujo import Flujo
from flujo.domain.models import PipelineResult, PipelineContext
from manual_testing.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE

# Ensure the API key is loaded from environment
def ensure_api_key():
    """Ensure the API key is loaded from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    print(f"✅ Using API key: {api_key[:20]}...")

async def main():
    # Ensure the correct API key is set
    ensure_api_key()

    print("Step 1: Running a single AI assessment step.\n" + "="*50)

    # The Flujo runner executes our pipeline.
    # We provide a pipeline_name and context_model to demonstrate that even
    # with a context present, the stateless agent works perfectly (FSD-11 fix).
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        pipeline_name="cohort_clarification_v1",
        context_model=PipelineContext
    )

    initial_definition = input("Enter the clinical cohort definition: ")

    # Because main() is an `async` function, we must use `run_async`.
    # It returns an async iterator; the last item is the final result.
    result: Optional[PipelineResult] = None
    async for item in runner.run_async(
        initial_definition,
        initial_context_data={"initial_prompt": initial_definition}
    ):
        result = item

    print("\n--- Agent Response ---")
    if result and result.step_history and result.step_history[-1].success:
        final_output = result.step_history[-1].output
        print(final_output)
    else:
        print("Pipeline execution failed.")
        if result and result.step_history:
            print(f"Feedback: {result.step_history[-1].feedback}")
        return # Exit if the run failed

    # --- FSD-12: TRACING DEMONSTRATION ---
    print("\n" + "="*50)
    print("✨ OBSERVABILITY (FSD-12) ✨")
    if result.final_pipeline_context and result.final_pipeline_context.run_id:
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
