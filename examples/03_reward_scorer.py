"""
03_reward_scorer.py
-------------------
Enable the experimental reward-model scorer (extra LLM judge).
"""

import os
from flujo import (
    Orchestrator,
    Task,
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
)
from flujo.infra.settings import settings

# 🔑 Make sure you have a paid API key – the reward model is another call
os.environ["REWARD_ENABLED"] = "true"
settings.scorer = "reward"

# Create orchestrator with the required agents
orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent)

best = orch.run_sync(Task(prompt="Summarise the Zen of Python in two sentences."))
print("Reward-model score:", best.score)
print("\nSolution:\n", best.solution)
