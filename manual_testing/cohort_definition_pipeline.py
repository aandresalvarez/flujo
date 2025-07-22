"""
Cohort Definition Clarification Pipeline (AI Agentic Version)
------------------------------------------------------------
This script implements a robust agentic pipeline that:
1. Receives a clinical cohort definition as input (text).
2. Uses an AI agent to check if the definition is clear enough.
3. If unclear, the agent asks for clarification (can be human-in-the-loop or LLM-based).
4. Iterates until the definition is clear, then returns the clarified, redefined cohort definition.

This is for manual testing and demonstration purposes.
"""

import asyncio
from flujo import Flujo, Step, Pipeline
from flujo.infra.agents import make_agent_async
from flujo.domain.models import PipelineContext

# --- CONFIGURATION ---
# Replace with your preferred model and system prompt
MODEL_NAME = "openai:gpt-4o"  # or another supported model
SYSTEM_PROMPT = (
    "You are a clinical research assistant. "
    "Your job is to review a clinical cohort definition provided by the user. "
    "If the definition is unclear, ambiguous, or missing key details, ask the user for clarification. "
    "If the definition is clear and complete, return the clarified definition and state that it is clear. "
    "Always be concise and specific in your requests for clarification."
)

# --- AGENT DEFINITION ---
# The agent receives the current cohort definition and returns either:
# - a clarified definition (if clear)
# - or a request for clarification (if unclear)

ClarificationAgent = make_agent_async(
    model=MODEL_NAME,
    system_prompt=SYSTEM_PROMPT,
    output_type=str,
    max_retries=2,
    timeout=60,
)

# --- LOOP STEP: Iteratively clarify until clear ---
# The agent is called repeatedly until the output contains a marker that the definition is clear.

CLEAR_MARKER = "[COHORT DEFINITION CLEAR]"

async def clarification_loop_step(definition: str, *, context: PipelineContext) -> str:
    """
    Calls the agent to clarify the cohort definition. If not clear, expects the user to clarify.
    """
    # Call the agent
    response = await ClarificationAgent.run(definition, context=context)
    # If the agent thinks the definition is clear, it should include the marker
    return response

# The loop body is a single agent step
loop_body_pipeline = Pipeline.from_step(
    Step.from_callable(clarification_loop_step, name="ClarificationAgentStep")
)

# The exit condition checks for the clear marker
exit_condition = lambda output, ctx: CLEAR_MARKER in output

clarification_loop = Step.loop_until(
    name="ClarificationLoop",
    loop_body_pipeline=loop_body_pipeline,
    exit_condition_callable=exit_condition,
    max_loops=5,
)

# --- FULL PIPELINE ---
full_pipeline = clarification_loop

runner = Flujo(full_pipeline, context_model=PipelineContext)

# --- MAIN EXECUTION ---
def main():
    print("Clinical Cohort Definition Clarification Pipeline (AI Agentic)\n" + "-"*60)
    initial_definition = input("Enter the initial clinical cohort definition: ").strip()
    # Run the pipeline asynchronously
    result = asyncio.run(runner.run(initial_definition))
    # The final output is in the last step's output
    final_output = result.step_history[-1].output
    print("\nFinal clarified cohort definition:")
    print(final_output.replace(CLEAR_MARKER, "").strip())

if __name__ == "__main__":
    main()
