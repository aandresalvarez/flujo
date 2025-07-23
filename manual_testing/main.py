# manual_testing/main.py
import traceback
from flujo import Flujo
from manual_testing.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE

def main():
    print("Step 1: Running a single AI assessment step.\n" + "="*50)

    # The Flujo runner executes our pipeline
    # Note: We don't need context for this simple pipeline
    runner = Flujo(COHORT_CLARIFICATION_PIPELINE, context_model=None)

    initial_definition = input("Enter the clinical cohort definition: ")

    try:
        # The run() method executes the pipeline from start to finish
        result = runner.run(initial_definition)

        print("\n--- Agent Response ---")
        if result and result.step_history:
            # Check each step for success/failure
            for i, step in enumerate(result.step_history):
                print(f"Step {i}: {step.name}")
                if step.success:
                    print(f"  Success: {step.output}")
                else:
                    print(f"  Failed: {step.feedback}")
                    print(f"  Attempts: {step.attempts}")

            # The final output is the output of the last step
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
    # Ensure you have OPENAI_API_KEY (or other) set in your .env file
    main()
