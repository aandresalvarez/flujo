"""
00_quickstart.py
----------------
Most basic usage: call the orchestrator once and print the result.
Run with:
    OPENAI_API_KEY=sk-... python 00_quickstart.py
"""

from flujo.recipes import Default
from flujo import (
    Task,
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
)

# 1️⃣  Create a default recipe (uses GPT-4o for all agents)
orch = Default(review_agent, solution_agent, validator_agent, reflection_agent)

# 2️⃣  Wrap your prompt in a Task (metadata optional)
task = Task(prompt="Write a short motivational haiku about debugging.")

# 3️⃣  Synchronous, blocking call – returns a Candidate object
best = orch.run_sync(task)

# 4️⃣  Inspect the result with null-safety
if best:
    print("Score:", best.score)
    print("\nSolution:\n", best.solution)
    print("\nChecklist:")
    if best.checklist:
        for item in best.checklist.items:
            print(f" • {item.description:<40}  =>  {item.passed}")
    else:
        print("  No checklist was generated for this solution.")
else:
    print("No solution was found.")
