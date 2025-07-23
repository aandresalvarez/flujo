"""
Cohort Definition Clarification Pipeline (AI Agentic Version)

This module defines a Flujo pipeline for iteratively clarifying a clinical
cohort definition. It uses an AI agent to assess clarity and, if needed,
integrates human-in-the-loop steps to request and incorporate clarifications.

The pipeline is designed for reusability and explicit state management via
a custom PipelineContext.
"""

from typing import Any
from flujo import Step, Pipeline, step
from flujo.infra.agents import make_agent_async
from flujo.domain.models import PipelineContext, BaseModel
from flujo.infra.settings import settings as flujo_settings
from pydantic import Field

# --- 1. Custom PipelineContext for explicit state management ---
class CohortContext(PipelineContext):
    """
    Custom context for the Cohort Definition Clarification Pipeline.
    Stores the current definition, clarity status, and human interaction history.
    """
    current_definition: str
    is_clear: bool = False
    clarification_requests_count: int = 0

# --- 2. Agent Configuration ---
CLARIFICATION_MODEL = flujo_settings.default_solution_model

CLARIFICATION_AGENT_SYSTEM_PROMPT = """
You are an expert clinical research assistant tasked with refining clinical cohort definitions.
Your goal is to ensure the provided definition is unambiguous, complete, and clinically sound.

**Input:** A clinical cohort definition.

**Instructions:**
1.  **Review:** Carefully read the provided cohort definition.
2.  **Assess Clarity:** Determine if the definition is sufficiently clear, unambiguous, and complete for a clinical study.
    *   Consider criteria such as: inclusion/exclusion criteria, time windows, patient characteristics, diagnosis codes, medication use, lab values, etc.
3.  **Action:**
    *   **If CLEAR:** Respond with the *entire, re-stated, and confirmed cohort definition*. Begin your response with the phrase `[CLARITY_CONFIRMED]`.
    *   **If UNCLEAR/AMBIGUOUS/INCOMPLETE:** Respond with a *single, precise question or request for specific additional information* needed to clarify the definition. Do NOT re-state the definition. Do NOT use the `[CLARITY_CONFIRMED]` marker.
    *   Be concise and professional in all responses.

**Example Clear Output:**
[CLARITY_CONFIRMED] Patients diagnosed with Type 2 Diabetes (ICD-10 E11.x) between January 1, 2020 and December 31, 2022, aged 18-65, with at least two HbA1c measurements > 7.0% within the first 6 months post-diagnosis, and no history of insulin use prior to diagnosis.

**Example Unclear Output:**
Please specify the exact ICD-10 codes for Type 2 Diabetes to be included.
"""

ClarificationAgent = make_agent_async(
    model=CLARIFICATION_MODEL,
    system_prompt=CLARIFICATION_AGENT_SYSTEM_PROMPT,
    output_type=str,
    max_retries=2,
    timeout=60,
)

# --- 3. Define Pipeline Steps using @step decorator ---

# IMPORTANT: This step directly modifies context, so do NOT use updates_context=True.
@step(name="AssessAndClarify", updates_context=True)
async def assess_and_clarify(definition_to_assess: str, *, context: CohortContext) -> dict:
    """
    Uses the ClarificationAgent and returns a dictionary containing context updates
    and the raw agent response for the next step.
    """
    agent_response = await ClarificationAgent.run(definition_to_assess, context=context)

    updates = {"agent_response": agent_response}  # Pass raw response to the router
    if agent_response.startswith("[CLARITY_CONFIRMED]"):
        updates["is_clear"] = True
        updates["current_definition"] = agent_response.replace("[CLARITY_CONFIRMED]", "").strip()
    else:
        updates["is_clear"] = False
        updates["clarification_requests_count"] = context.clarification_requests_count + 1

    return updates

class HumanClarificationInput(BaseModel):
    """
    Schema for human input providing clarification to the cohort definition.
    """
    clarification: str = Field(
        ..., description="The clarification or additional information provided by the human."
    )

@step(name="IncorporateHumanClarification", updates_context=True)
async def incorporate_human_clarification(
    human_response: HumanClarificationInput, *, context: CohortContext
) -> dict:
    """
    Combines the current definition (from context) with human-provided clarification.
    Returns a dict to update context.current_definition.
    """
    combined_definition = f"{context.current_definition}\n\nHuman's clarification: {human_response.clarification}"
    return {"current_definition": combined_definition}

# --- 4. Define the Loop Structure ---
def is_clarification_needed(data: dict, context: CohortContext) -> str:
    # The context is updated by the previous `assess_and_clarify` step
    return "DEFINITION_CLEAR" if context.is_clear else "CLARIFICATION_NEEDED"

human_in_the_loop_step = Step.human_in_the_loop(
    name="HumanInputForClarification",
    message_for_user=(
        "The AI agent needs clarification. Please provide additional details:\n"
        "AI Request: {input[agent_response]}"
    ),
    input_schema=HumanClarificationInput,
)

branch_clear_pipeline = Pipeline.from_step(
    Step.from_callable(lambda x: x, name="DefinitionConfirmed")
)

branch_clarify_pipeline = human_in_the_loop_step >> incorporate_human_clarification

check_clarity_and_route = Step.branch_on(
    name="CheckClarityAndRoute",
    condition_callable=is_clarification_needed,
    branches={
        "DEFINITION_CLEAR": branch_clear_pipeline,
        "CLARIFICATION_NEEDED": branch_clarify_pipeline,
    },
)

loop_body = assess_and_clarify >> check_clarity_and_route

def loop_exit_condition(output: Any, context: CohortContext) -> bool:
    return context.is_clear

clarification_loop = Step.loop_until(
    name="CohortClarificationLoop",
    loop_body_pipeline=loop_body,
    exit_condition_callable=loop_exit_condition,
    max_loops=10,
    initial_input_to_loop_body_mapper=lambda initial_prompt_from_runner, ctx: ctx.current_definition,
    iteration_input_mapper=lambda last_output_of_loop_body, ctx, i: ctx.current_definition,
    loop_output_mapper=lambda last_output_of_loop_body, ctx: ctx.current_definition,
)

COHORT_CLARIFICATION_PIPELINE = Pipeline.from_step(clarification_loop)
