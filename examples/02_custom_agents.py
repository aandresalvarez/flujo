"""
Demonstrates creating custom agents with specific personalities and
instructions, then using them with the high-level pipeline factory.
"""

import asyncio
from flujo import make_agent_async, init_telemetry, Task
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Checklist

init_telemetry()


async def main():
    print("🎨 Creating a team of custom, specialized agents for a creative task...")

    # A custom reviewer focused on the specific structure of a limerick.
    limerick_reviewer = make_agent_async(
        "openai:gpt-4o",
        "You are a poetry critic. Create a checklist to verify a limerick's AABBA rhyme scheme and rhythm. Be specific.",
        Checklist,
    )

    # A custom solution agent with a personality. A cheaper model is fine for creative tasks.
    limerick_writer = make_agent_async(
        "openai:gpt-4o-mini",
        "You are a witty and slightly mischievous poet. Write a funny limerick based on the user's topic.",
        str,
    )

    # A custom validator that embodies the role of a strict poetry judge.
    limerick_validator = make_agent_async(
        "openai:gpt-4o",
        "You are a strict poetry judge. Use the provided checklist to rigorously grade the limerick. Do not be lenient.",
        Checklist,
    )

    # Create the pipeline using the factory
    pipeline = make_default_pipeline(
        review_agent=limerick_reviewer,
        solution_agent=limerick_writer,
        validator_agent=limerick_validator,
    )

    task = Task(prompt="Write a limerick about a robot who discovers coffee.")

    print("🧠 Running workflow with custom agents...")
    result = await run_default_pipeline(pipeline, task)

    if result:
        print("\n🎉 Workflow finished!")
        print("-" * 50)
        print(f"The winning limerick:\n\n{result.solution}\n")

        if result.checklist:
            print("Quality Checklist:")
            for item in result.checklist.items:
                status = "✅ Passed" if item.completed else "❌ Failed"
                print(f"  - {item.description:<60} {status}")
    else:
        print("\n❌ The workflow did not produce a valid solution.")


if __name__ == "__main__":
    asyncio.run(main())
