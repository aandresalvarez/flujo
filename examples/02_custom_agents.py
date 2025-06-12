"""
02_custom_agents.py
-------------------
Swap GPT-4o for a cheaper GPT-3.5-turbo *just* for the solution agent
to save tokens.
"""

from pydantic_ai import Agent
from pydantic_ai_orchestrator import Orchestrator
from pydantic_ai_orchestrator.infra.agents import review_agent, validator_agent, get_reflection_agent

# Build a single cheaper agent
solution_agent = Agent(
    "openai:gpt-3.5-turbo",
    system_prompt="You are an efficient programmer – output concise, correct code.",
    output_type=str,
)

orch = Orchestrator(
    review_agent=review_agent,          # keep the "review" step on GPT-4o
    solution_agent=solution_agent,      # cheaper generation
    validator_agent=validator_agent,
    reflection_agent=get_reflection_agent(),
    max_iters=2,
    k_variants=1,
)

print(orch.run_sync("Write a limerick that scans.").solution) 