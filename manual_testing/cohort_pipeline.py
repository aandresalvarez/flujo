# manual_testing/cohort_pipeline.py

from flujo import Step, Pipeline
from flujo.infra.agents import make_agent_async
from flujo.infra.settings import settings as flujo_settings

# Define the AI agent that will assess the cohort definition
CLARIFICATION_AGENT_SYSTEM_PROMPT = """
You are a clinical research assistant. Your job is to review a clinical cohort definition.
If the definition is clear, re-state it and add the marker '[CLARITY_CONFIRMED]'.
If the definition is unclear, ask a concise question to clarify it.
"""

# With FSD-11 fixed, make_agent_async is the simple, correct way to create
# a stateless agent. It will work perfectly.
ClarificationAgent = make_agent_async(
    model=flujo_settings.default_solution_model,
    system_prompt=CLARIFICATION_AGENT_SYSTEM_PROMPT,
    output_type=str,
)

# Create a single step that uses our agent
assess_clarity_step = Step(
    name="AssessClarity",
    agent=ClarificationAgent,
)

# The pipeline consists of just this one step
COHORT_CLARIFICATION_PIPELINE = Pipeline.from_step(assess_clarity_step)
