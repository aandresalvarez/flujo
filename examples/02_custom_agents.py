"""
02_custom_agents.py
-------------------
Swap GPT-4o for a cheaper GPT-3.5-turbo *just* for the solution agent
to save tokens.
"""

from pydantic_ai import Agent
from flujo import (
    Orchestrator,
    Task,
    review_agent,
    validator_agent,
    reflection_agent,
)

# Build a single cheaper agent
solution_agent = Agent(
    "openai:gpt-4.1-nano",
    system_prompt="You are an efficient programmer – output concise, correct code.",
    output_type=str,
)

orch = Orchestrator(
    review_agent=review_agent,  # keep the "review" step on GPT-4o
    solution_agent=solution_agent,  # cheaper generation
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
    max_iters=2,
    k_variants=1,
)

# Wrap the prompt in a Task object
task = Task(prompt="Write a limerick that scans.")
print(orch.run_sync(task).solution)
