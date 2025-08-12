"""
Demonstrates using the pipeline factory with SQL validation plugin.
This example shows how to add custom validation logic to your workflows.
For more details, see docs/extending.md.
"""

import asyncio
from flujo import make_agent_async, init_telemetry, Task
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Checklist
from flujo.plugins.sql_validator import SQLSyntaxValidator

init_telemetry()


async def main():
    print("🧠 Running a SQL generation and validation pipeline...")

    # Create agents for the pipeline
    review_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a SQL expert. Create a checklist for writing correct SQL queries.",
        Checklist,
    )

    solution_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a SQL developer. Write SQL queries that meet the requirements.",
        str,
    )

    validator_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a SQL validator. Check if the query meets the requirements.",
        Checklist,
    )

    # Create the pipeline using the factory
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Add SQL validation plugin to the solution step
    # This will validate SQL syntax before the validator agent runs
    pipeline.steps[1].add_plugin(SQLSyntaxValidator())

    # Run the pipeline
    task = Task(prompt="Write a SQL query to select all users from a table called 'users'")
    result = await run_default_pipeline(pipeline, task)

    if result:
        print("\n✅ SQL Query Generated!")
        print("-" * 50)
        print(f"Query:\n{result.solution}")

        if result.checklist:
            print("\nValidation Checklist:")
            for item in result.checklist.items:
                status = "✅ Passed" if item.completed else "❌ Failed"
                print(f"  - {item.description:<60} {status}")
    else:
        print("\n❌ Failed to generate a valid SQL query.")


if __name__ == "__main__":
    asyncio.run(main())
