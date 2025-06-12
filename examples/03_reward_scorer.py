"""
03_reward_scorer.py
-------------------
Enable the experimental reward-model scorer (extra LLM judge).
"""

import os
from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.settings import settings

# 🔑 Make sure you have a paid API key – the reward model is another call
os.environ["REWARD_ENABLED"] = "true"
settings.scorer = "reward"

best = Orchestrator().run_sync(Task(prompt="Summarise the Zen of Python in two sentences."))
print("Reward-model score:", best.score)
print(best.solution) 