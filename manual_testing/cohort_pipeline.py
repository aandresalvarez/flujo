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

# Create a simple mock agent to test signature inspection
class MockAgent:
    """A simple agent that doesn't accept context parameter"""
    def __init__(self, name="MockAgent"):
        self.name = name

    async def run(self, data: str) -> str:
        """This method does NOT accept a context parameter - this is the bug case"""
        return f"Mock response to: {data}"

# Create a context-aware mock agent for comparison
class ContextAwareMockAgent:
    """A context-aware agent that accepts context parameter"""
    def __init__(self, name="ContextAwareMockAgent"):
        self.name = name

    async def run(self, data: str, context=None) -> str:
        """This method DOES accept a context parameter - this should work"""
        context_info = f" (context: {type(context).__name__})" if context else ""
        return f"Context-aware response to: {data}{context_info}"

# Use the context-aware mock agent for testing
ClarificationAgent = ContextAwareMockAgent("TestContextAwareAgent")

# Create a single step that uses our agent
assess_clarity_step = Step(
    name="AssessClarity",
    agent=ClarificationAgent,
)

# The pipeline consists of just this one step
COHORT_CLARIFICATION_PIPELINE = Pipeline.from_step(assess_clarity_step)
