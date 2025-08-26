"""
A practical example of processing a batch of prompts concurrently.
This pattern is highly efficient and leverages Python's `asyncio.gather`
to run multiple `flujo` workflows at the same time.
"""

import asyncio
from flujo import make_agent_async, init_telemetry, Task
from flujo.recipes.factories import make_default_pipeline, run_default_pipeline
from flujo.domain.models import Checklist

init_telemetry()


async def main():
    # Create agents for the pipeline
    review_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a code reviewer. Create a checklist for the given programming task.",
        Checklist,
    )

    solution_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a Python developer. Write code that meets the requirements.",
        str,
    )

    validator_agent = make_agent_async(
        "openai:gpt-4o",
        "You are a validator. Check if the solution meets the requirements.",
        Checklist,
    )

    # Create the pipeline using the factory
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Define multiple tasks to process
    tasks = [
        Task(prompt="Write a function to calculate factorial"),
        Task(prompt="Write a function to reverse a string"),
        Task(prompt="Write a function to check if a number is prime"),
        Task(prompt="Write a function to find the maximum in a list"),
    ]

    print("🚀 Processing multiple tasks concurrently...")

    # Process all tasks concurrently
    results = await asyncio.gather(
        *[run_default_pipeline(pipeline, task) for task in tasks], return_exceptions=True
    )

    # Display results
    for i, result in enumerate(results):
        print(f"\n--- Task {i + 1}: {tasks[i].prompt} ---")
        if isinstance(result, Exception):
            print(f"❌ Error: {result}")
        elif result:
            print(f"✅ Solution: {result.solution[:100]}...")
        else:
            print("❌ No result produced")


if __name__ == "__main__":
    asyncio.run(main())
