"""
Demonstrates using weighted scoring to prioritize certain quality criteria.
For more details on scoring, see docs/scoring.md.
"""
from flujo.recipes import Default
from flujo import (
    Task,
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
    init_telemetry,
)

init_telemetry()

# Scenario: We want a Python function, but we consider having a good
# docstring (for maintainability) more important than using type hints.
# We can express this preference with weighted scoring.
weights = [
    {"item": "Includes a comprehensive docstring", "weight": 0.7},
    {"item": "Uses type hints for all parameters and return values", "weight": 0.3},
]

# The weights are passed via the Task's `metadata` dictionary.
# The `Default` recipe will automatically detect these weights and use the
# `weighted_score` function if the `scorer` setting is 'weighted'.
task = Task(
    prompt="Write a Python function that adds two numbers. It must have a docstring and type hints.",
    metadata={"weights": weights},
)

# If your global settings have `scorer` as 'ratio', you can override it
# in the metadata as well: `metadata={"weights": weights, "scorer": "weighted"}`
orch = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
)

print("🧠 Running workflow with weighted scoring (prioritizing docstrings)...")
best_candidate = orch.run_sync(task)

if best_candidate:
    print("\n🎉 Workflow finished!")
    print("-" * 50)
    print(f"Solution:\n{best_candidate.solution}")
    print(f"\nWeighted Score: {best_candidate.score:.2f}")
    if best_candidate.checklist:
        print("\nFinal Quality Checklist:")
        for item in best_candidate.checklist.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            print(f"  - {item.description:<60} {status}")
else:
    print("\n❌ The workflow did not produce a valid solution.")
