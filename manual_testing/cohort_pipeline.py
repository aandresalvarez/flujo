# manual_testing/cohort_pipeline.py

import asyncio
from flujo import Step, Pipeline
from flujo.infra.agents import make_agent
from flujo.infra.settings import settings as flujo_settings

# Define the AI agent that will assess the cohort definition
CLARIFICATION_AGENT_SYSTEM_PROMPT = """
You are a clinical research assistant. Your job is to review a clinical cohort definition.
If the definition is clear, re-state it and add the marker '[CLARITY_CONFIRMED]'.
If the definition is unclear, ask a concise question to clarify it.
"""

# Create the agent directly without the async wrapper
ClarificationAgent, _ = make_agent(
    model=flujo_settings.default_solution_model,
    system_prompt=CLARIFICATION_AGENT_SYSTEM_PROMPT,
    output_type=str,
)

# Create a simple wrapper that removes context parameter
class SimpleAgentWrapper:
    def __init__(self, agent):
        self.agent = agent

    async def run(self, data, **kwargs):
        # Remove context and other unwanted parameters
        clean_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ['context', 'pipeline_context', 'resources']}
        return await self.agent.run(data, **clean_kwargs)

# Create a single step that uses our wrapped agent
assess_clarity_step = Step(
    name="AssessClarity",
    agent=SimpleAgentWrapper(ClarificationAgent),
    persist_feedback_to_context=None,  # Disable context persistence
)

# The pipeline consists of just this one step
COHORT_CLARIFICATION_PIPELINE = Pipeline.from_step(assess_clarity_step)
