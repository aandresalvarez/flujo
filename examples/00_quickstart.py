"""
A basic "Hello, World!" example showing the high-level Default recipe.
This is the fastest way to get a multi-agent workflow running.
"""
from flujo.recipes import Default
from flujo import (
    Task,
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,  # It's a best practice to include the reflection agent.
    init_telemetry,
)

# It's good practice to initialize telemetry at the start of your application.
# This configures logging and tracing. See docs/telemetry.md for more info.
init_telemetry()

# The Default recipe is a pre-built workflow for a common multi-agent pattern:
# Review -> Solution -> Validate -> Reflection.
# It's great for getting started quickly without building a custom pipeline.
print("🤖 Assembling the AI agent team for the standard Default workflow...")
orch = Default(
    review_agent=review_agent,
    solution_agent=solution_agent,
    validator_agent=validator_agent,
    reflection_agent=reflection_agent,
)

task = Task(prompt="Write a Python function that returns the string 'Hello, World!'")

print("🧠 Running the workflow...")
best_candidate = orch.run_sync(task)

# The `best_candidate` object contains the final solution, its score, and
# the checklist used to evaluate it.
if best_candidate:
    print("\n🎉 Workflow finished! Here is the best result:")
    print("-" * 50)
    print(f"Solution:\n{best_candidate.solution}")
    print(f"\nQuality Score: {best_candidate.score:.2f}")
    if best_candidate.checklist:
        print("\nFinal Quality Checklist:")
        for item in best_candidate.checklist.items:
            status = "✅ Passed" if item.passed else "❌ Failed"
            # The description is left-aligned to 50 characters for clean printing.
            print(f"  - {item.description:<50} {status}")
else:
    print("\n❌ The workflow did not produce a valid solution.")
