# manual_testing/cohort_pipeline.py

from flujo import Step, Pipeline, step
from flujo.infra.agents import make_agent_async
from flujo.domain.models import PipelineContext
from flujo.infra.settings import settings as flujo_settings

# --- STEP 3: INTRODUCING STATE WITH PIPELINECONTEXT ---
#
# This step introduces the core state management concept in Flujo:
# - Custom PipelineContext: Creating a Pydantic model to store state
# - @step(updates_context=True): Steps that can modify the shared context
# - runner.run(initial_context_data=...): Starting a run with initial state
# - Mappers in Step.loop_until: Controlling data flow between loop iterations
#
# This fixes the limitation from Step 2 where the agent kept asking about
# the original prompt instead of the updated one.

# Define a custom context to store the evolving state of our cohort definition
class CohortContext(PipelineContext):
    """
    Custom PipelineContext that stores the evolving state of our cohort definition.

    This is the key concept: instead of passing data between steps as function
    parameters, we store it in a shared context that all steps can access and modify.
    """
    current_definition: str  # The evolving definition
    is_clear: bool = False   # Flag indicating if the definition is clear
    clarification_count: int = 0  # Track how many clarifications we've requested
    # Note: initial_prompt is inherited from PipelineContext and is required

# Define the AI agent that will assess the cohort definition
CLARIFICATION_AGENT_SYSTEM_PROMPT = """
You are a clinical research assistant. You will be given the current state of a cohort definition.
If it is clear, re-state it and add the marker '[CLARITY_CONFIRMED]'.
If it is unclear, ask a *single* clarifying question.
"""

# With FSD-11 fixed, make_agent_async is the simple, correct way to create
# a stateless agent. It will work perfectly.
ClarificationAgent = make_agent_async(
    model=flujo_settings.default_solution_model,
    system_prompt=CLARIFICATION_AGENT_SYSTEM_PROMPT,
    output_type=str,
)

# --- NEW: A step that updates the context ---
@step(name="AssessAndRefine", updates_context=True)
async def assess_and_refine(definition_to_assess: str, *, context: CohortContext) -> dict:
    """
    This step calls the agent and returns a dictionary of updates for the context.

    The @step(updates_context=True) decorator tells Flujo that this step can
    modify the shared context. The return value is a dictionary of field updates.

    Args:
        definition_to_assess: The definition to assess (from the loop input)
        context: The shared CohortContext that all steps can access

    Returns:
        dict: Updates to apply to the context
    """
    # Debug logging removed for production

    # Call the agent with the current definition
    agent_response = await ClarificationAgent.run(definition_to_assess, context=context)

    # Extract the actual output from the agent response
    # The agent might return an object with an 'output' attribute or a string directly
    agent_output = getattr(agent_response, "output", agent_response)

    # Debug logging removed for production

    if "[CLARITY_CONFIRMED]" in agent_output:
        # If clear, update the definition and set the 'is_clear' flag to True
        final_definition = agent_output.replace("[CLARITY_CONFIRMED]", "").strip()
        updates = {
            "current_definition": final_definition,
            "is_clear": True
        }
        # Debug logging removed for production
        return updates
    else:
        # If not clear, simulate human clarification based on the agent's question
        # This is a simplified simulation for Step 3 demonstration.
        clarification = _simulate_human_clarification(agent_output, context.clarification_count)

        # Combine the old definition with the clarification for the next round
        new_definition = f"Definition so far: '{context.current_definition}'. Clarification: '{clarification}'"
        updates = {
            "current_definition": new_definition,
            "is_clear": False,
            "clarification_count": context.clarification_count + 1
        }
        # Debug logging removed for production
        return updates

def _simulate_human_clarification(agent_question: str, clarification_count: int) -> str:
    """
    Simulate human clarifications based on the agent's questions.
    This is a simplified simulation for Step 3 demonstration.
    """
    question_lower = agent_question.lower()

    if "age" in question_lower or "time frame" in question_lower:
        return "patients of all ages, seen in the last 12 months"
    elif "medication" in question_lower and "currently" in question_lower:
        return "patients currently on asthma medication"
    elif "medication" in question_lower or "specific" in question_lower:
        return "any asthma medication, including inhalers and oral medications"
    elif "clinic" in question_lower and ("type" in question_lower or "specific" in question_lower):
        return "any outpatient clinic, including primary care and specialist visits"
    elif "clinic" in question_lower or "healthcare" in question_lower:
        return "any outpatient clinic, including primary care and specialist visits"
    elif "diagnosis" in question_lower or "currently" in question_lower:
        return "patients with confirmed asthma diagnosis, currently on medication"
    elif "severity" in question_lower or "type" in question_lower:
        return "any severity level, including mild, moderate, and severe asthma"
    else:
        # Default clarification for any other question
        return "all patients meeting the basic criteria"

# --- UPDATED: Loop now uses the context ---
# The body of our loop is now the context-updating step
loop_body_pipeline = Pipeline.from_step(assess_and_refine)

def exit_loop_when_clear(output: dict, context: CohortContext) -> bool:
    """
    The loop now checks the `is_clear` flag in our context.

    This is much more robust than parsing strings - we have explicit state!
    """
    print(f"  [Loop Check] Context 'is_clear' flag is: {context.is_clear}")
    return context.is_clear

def map_context_to_input(initial_input: str, context: CohortContext) -> str:
    """
    The first loop iteration should use the definition from the context.

    This mapper controls what data flows into the loop body on the first iteration.
    """
    return context.current_definition

def map_output_to_next_input(last_output: dict, context: CohortContext, i: int) -> str:
    """
    Subsequent iterations also use the updated definition from the context.

    This mapper controls what data flows into the loop body on subsequent iterations.
    """
    # Use the last output's current_definition if available, otherwise fall back to context
    if isinstance(last_output, dict) and "current_definition" in last_output:
        return last_output["current_definition"]
    return context.current_definition

# Create the LoopStep with context-aware mappers
clarification_loop = Step.loop_until(
    name="StatefulClarificationLoop",
    loop_body_pipeline=loop_body_pipeline,
    exit_condition_callable=exit_loop_when_clear,
    max_loops=5,
    # NEW: Mappers control the data flow
    initial_input_to_loop_body_mapper=map_context_to_input,
    iteration_input_mapper=map_output_to_next_input
)

# The pipeline is now just the loop, but with proper state management
COHORT_CLARIFICATION_PIPELINE = Pipeline.from_step(clarification_loop)
