# manual_testing/main.py
import traceback
from flujo import Flujo
from flujo.domain.models import PipelineContext # Import this for the runner
from manual_testing.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE

def main():
    print("Step 1: Running a single AI assessment step.\n" + "="*50)

    # The Flujo runner executes our pipeline.
    # CRITICAL: We pass a context_model to trigger the bug.
    # The runner will try to pass a context even if the agent doesn't want it.
    runner = Flujo(
        COHORT_CLARIFICATION_PIPELINE,
        context_model=PipelineContext, # This will trigger the context injection
        pipeline_name="test_bug_run",
        initial_context_data={"initial_prompt": "Test prompt"}  # Provide required field
    )

    initial_definition = input("Enter the clinical cohort definition: ")

    try:
        # The run() method executes the pipeline from start to finish
        result = runner.run(initial_definition)

        print("\n--- Agent Response ---")
        if result and result.step_history:
            final_step = result.step_history[-1]
            if final_step.success:
                print(f"\nFinal Output: {final_step.output}")
            else:
                print(f"\nPipeline failed: {final_step.feedback}")
        else:
            print("Pipeline execution failed - no step history.")
    except Exception as e:
        print(f"Pipeline execution failed with exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
