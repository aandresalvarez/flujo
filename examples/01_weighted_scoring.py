"""
Demonstrates using weighted scoring to prioritize certain quality criteria.
For more details on scoring, see docs/scoring.md.
"""
from flujo.recipes import Default
from flujo import (
    Task,
    make_agent_async,
    solution_agent,
    validator_agent,
    reflection_agent,
    init_telemetry,
)
from flujo.infra.settings import settings
from flujo.domain.models import Checklist

init_telemetry()

# Switch scoring to weighted so the orchestrator honors our weights.
settings.scorer = "weighted"

# Define our checklist items and their relative importance.
weights = [
    {"item": "Includes a docstring", "weight": 0.7},
    {"item": "Uses type hints", "weight": 0.3},
]

# Create a custom review agent that returns exactly these checklist items.
REVIEW_SYS = """You are an expert Python reviewer. Provide a checklist with these exact items:\n1. \"Includes a docstring\"\n2. \"Uses type hints\"\nReturn JSON only matching Checklist model."""
review_agent = make_agent_async(settings.default_review_model, REVIEW_SYS, Checklist)

# Our task asks for a simple addition function with both a docstring and type hints.
task = Task(
    prompt="Write a Python function that adds two numbers using type hints and a clear docstring.",
    metadata={"weights": weights},
)

# The Default recipe will automatically apply the weighted_score function now that
# settings.scorer is set to 'weighted' and weights are provided via metadata.
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
