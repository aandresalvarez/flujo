"""
Demonstrates using weighted scoring to prioritize certain quality criteria.
For more details on scoring, see docs/scoring.md.
"""

import asyncio
from flujo import make_agent_async, init_telemetry, Task
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Checklist
from flujo.domain.scoring import weighted_score

init_telemetry()


async def main():
    # Create agents for the pipeline
    review_agent = make_agent_async(
        "openai:gpt-4o",
        "You are an expert Python reviewer. Create a checklist with these items:\n1. 'Includes a docstring'\n2. 'Uses type hints'\n3. 'Function works correctly'",
        Checklist,
    )

    solution_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a Python developer. Write code that meets the requirements.",
        str,
    )

    validator_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a validator. For each checklist item, explicitly set 'passed' to true if the solution meets the requirement, or false if it does not. Return a Checklist with the same items, each with a boolean 'passed' field.",
        Checklist,
    )

    # Create the pipeline using the factory
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Run the pipeline
    task = Task(
        prompt="Write a Python function that adds two numbers using type hints and a clear docstring."
    )
    result = await run_default_pipeline(pipeline, task)

    if result:
        print("\n🎉 Workflow finished!")
        print("-" * 50)
        print(f"Solution:\n{result.solution}")

        # Use weighted scoring to prioritize docstrings
        weights = [
            {"item": "Includes a docstring", "weight": 2.0},  # Double weight for docstrings
            {"item": "Uses type hints", "weight": 1.5},  # Higher weight for type hints
            {"item": "Function works correctly", "weight": 1.0},
        ]

        score = weighted_score(result.checklist, weights)
        print(f"\nWeighted Score: {score:.2f}")

        if result.checklist:
            print("\nFinal Quality Checklist:")
            for item in result.checklist.items:
                status = "✅ Passed" if item.passed else "❌ Failed"
                print(f"  - {item.description:<60} {status}")
    else:
        print("\n❌ The workflow did not produce a valid solution.")


if __name__ == "__main__":
    asyncio.run(main())
