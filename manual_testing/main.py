"""
CLI for running the Cohort Definition Clarification Pipeline using Flujo.
Handles agentic and human-in-the-loop steps interactively.
"""
import asyncio
from typing import Optional
from flujo import Flujo
from flujo.domain.models import PipelineResult
from manual_testing.cohort_pipeline import COHORT_CLARIFICATION_PIPELINE, CohortContext, HumanClarificationInput

def main():
    """
    Run the cohort clarification pipeline interactively via CLI.
    Handles PausedException for human-in-the-loop steps and resumes as needed.
    """
    print("Flujo Clinical Cohort Definition Clarification (Agentic & Human-in-the-Loop)\n" + "="*80)

    runner = Flujo(COHORT_CLARIFICATION_PIPELINE, context_model=CohortContext)
    run_id = "cohort-clarification-run-123"
    initial_definition = (
        "Patients with asthma, on medication, seen in clinic."
    )
    # CRUCIAL: Initialize current_definition in initial_context_data for robust context
    initial_context_data = {
        "run_id": run_id,
        "initial_prompt": initial_definition,
        "current_definition": initial_definition,
    }
    print(f"\n--- Initial Cohort Definition (Run ID: {run_id}) ---")
    print(initial_definition)
    print("-" * 30)

    result: Optional[PipelineResult[CohortContext]] = None
    while True:
        try:
            # Run the pipeline. If it pauses, PausedException will be raised.
            result = runner.run(initial_definition, run_id=run_id, initial_context_data=initial_context_data)
            break  # Pipeline completed
        except Exception as e:
            from flujo.exceptions import PausedException
            if isinstance(e, PausedException):
                print(f"\n--- PAUSED FOR HUMAN INPUT ---")
                print(f"AI Agent Request: {e.message}")
                human_input_text = input("Your clarification: ").strip()
                if result is None:
                    print("Error: Pipeline paused but no partial result available to resume from.")
                    break
                human_input_model = HumanClarificationInput(clarification=human_input_text)
                # Resume pipeline asynchronously for human input
                result = asyncio.run(runner.resume_async(result, human_input_model))
            else:
                print(f"\n--- PIPELINE FAILED ---")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                break
    print("\n" + "="*80)
    if result and result.final_pipeline_context:
        final_context = result.final_pipeline_context
        print(f"Pipeline Run ID: {final_context.run_id}")
        print(f"Final Status: {final_context.scratchpad.get('status', 'unknown')}")
        print(f"Total Clarification Requests: {final_context.clarification_requests_count}")
        print(f"Is Definition Clear: {final_context.is_clear}")
        if final_context.is_clear:
            print("\n--- FINAL CLARIFIED COHORT DEFINITION ---")
            print(final_context.current_definition)
        else:
            print("\n--- PIPELINE DID NOT REACH CLARITY ---")
            print("Last definition status:")
            print(result.step_history[-1].output if result.step_history else "No output.")
        print("\n--- FULL PIPELINE HISTORY ---")
        for sr in result.step_history:
            print(f"- {sr.name}: Success={sr.success}, Output (truncated)={str(sr.output)[:100]}, Feedback={sr.feedback or 'N/A'}")
    else:
        print("Pipeline execution did not yield a result.")

if __name__ == "__main__":
    main()
