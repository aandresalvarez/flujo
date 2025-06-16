"""
Demonstrates creating custom agents with specific personalities and
instructions, then using them with the high-level Default recipe.
"""
from flujo.recipes import Default
from flujo import (
    Task,
    make_agent_async,
    reflection_agent,
    init_telemetry,
)
from flujo.domain.models import Checklist

init_telemetry()

# Scenario: We want a fun, creative output (a limerick). For this, generic
# agents might be too bland. We'll create custom agents with specialized
# system prompts to give them personality and focus.

print("🎨 Creating a team of custom, specialized agents for a creative task...")

# A custom reviewer focused on the specific structure of a limerick.
limerick_reviewer = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are a poetry critic. Create a checklist to verify a limerick's AABBA rhyme scheme and rhythm. Be specific.",
    output_type=Checklist,
)

# A custom solution agent with a personality. A cheaper model is fine for creative tasks.
limerick_writer = make_agent_async(
    model="openai:gpt-4o-mini",
    system_prompt="You are a witty and slightly mischievous poet. Write a funny limerick based on the user's topic.",
    output_type=str,
)

# A custom validator that embodies the role of a strict poetry judge.
limerick_validator = make_agent_async(
    model="openai:gpt-4o",
    system_prompt="You are a strict poetry judge. Use the provided checklist to rigorously grade the limerick. Do not be lenient.",
    output_type=Checklist,
)

# We can plug our custom agents directly into the `Default` recipe's workflow.
orch = Default(
    review_agent=limerick_reviewer,
    solution_agent=limerick_writer,
    validator_agent=limerick_validator,
    reflection_agent=reflection_agent,  # We can still use a default agent here.
)

task = Task(prompt="Write a limerick about a robot who discovers coffee.")

print("🧠 Running workflow with custom agents...")
best_candidate = orch.run_sync(task)

if best_candidate:
    print("\n🎉 Workflow finished!")
    print("-" * 50)
    print(f"The winning limerick:\n\n{best_candidate.solution}\n")
    print(f"Final Score: {best_candidate.score:.2f}")
else:
    print("\n❌ The workflow did not produce a valid solution.")
