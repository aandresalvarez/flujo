"""
01_weighted_scoring.py
----------------------
Demonstrates **weighted** checklist scoring.  Two items, different weights.
"""

from pydantic_ai_orchestrator import Orchestrator, Task
from pydantic_ai_orchestrator.infra.settings import settings

# 📝 Switch to weighted scoring – you can also set this in .env
settings.scorer = "weighted"

weights = [
    {"item": "Includes a docstring", "weight": 0.7},
    {"item": "Uses type hints",      "weight": 0.3},
]

task = Task(
    prompt="Write a Python function that reverses a string.",
    metadata={"weights": weights},
)

best = Orchestrator().run_sync(task)

print("Weighted score:", best.score)
for item in best.checklist.items:
    print(f"{item.description:<25} passed={item.passed}  weight={next((w['weight'] for w in weights if w['item']==item.description), 1.0)}") 