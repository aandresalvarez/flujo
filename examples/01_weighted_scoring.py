"""
01_weighted_scoring.py
----------------------
Demonstrates **weighted** checklist scoring.  Two items, different weights.
"""

from flujo.recipes import Default
from flujo import (
    Task,
    make_agent_async,
    solution_agent,
    validator_agent,
    reflection_agent,
)
from flujo.infra.settings import settings
from flujo.domain.models import Checklist

# 📝 Switch to weighted scoring – you can also set this in .env
settings.scorer = "weighted"

weights = [
    {"item": "Includes a docstring", "weight": 0.7},
    {"item": "Uses type hints", "weight": 0.3},
]

# Create a custom review agent with our specific criteria
CUSTOM_REVIEW_SYS = """You are an expert software engineer.
Your task is to generate a checklist of criteria to evaluate a solution for the user's request.
The checklist MUST include EXACTLY these items (copy them verbatim):
1. "Includes a docstring"
2. "Uses type hints"

Return **JSON only** that conforms to this schema:
Checklist(items=[ChecklistItem(description:str, passed:bool|None, feedback:str|None)])

Example:
{
  "items": [
    {"description": "Includes a docstring", "passed": null, "feedback": null},
    {"description": "Uses type hints", "passed": null, "feedback": null}
  ]
}
"""

review_agent = make_agent_async(settings.default_review_model, CUSTOM_REVIEW_SYS, Checklist)

task = Task(
    prompt="Write a Python function that reverses a string.",
    metadata={"weights": weights},
)

# Create the default recipe with the required agents
orch = Default(review_agent, solution_agent, validator_agent, reflection_agent)

best = orch.run_sync(task)

print("\nSolution:\n", best.solution)
print("\nWeighted score:", best.score)
print("\nChecklist:")
if best.checklist:
    for item in best.checklist.items:
        weight = next((w["weight"] for w in weights if w["item"] == item.description), 1.0)
        print(f" • {item.description:<25} passed={item.passed}  weight={weight:.1f}")
else:
    print("  No checklist was generated for this solution.")
