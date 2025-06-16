"""
A practical example of processing a batch of prompts concurrently.
This pattern is highly efficient and leverages Python's `asyncio.gather`
to run multiple `flujo` workflows at the same time.
"""
import asyncio
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


async def main():
    # A list of prompts we want to process in a batch.
    prompts = [
        "Write a tagline for a new brand of sparkling water.",
        "Generate a two-sentence horror story.",
        "Create a simple Python function to reverse a string.",
    ]

    # We use the same `Default` recipe instance for all tasks.
    orch = Default(
        review_agent, solution_agent, validator_agent, reflection_agent
    )

    # Create a list of asyncio tasks. Each task is a call to `run_async`.
    # This prepares the workflows but doesn't run them yet.
    print(f"🚀 Preparing a batch of {len(prompts)} workflows to run concurrently...")
    tasks_to_run = [
        orch.run_async(Task(prompt=p)) for p in prompts
    ]

    # `asyncio.gather` runs all the prepared tasks concurrently.
    # This is much faster than running them one by one in a loop.
    results = await asyncio.gather(*tasks_to_run)

    print("\n✅ Batch processing complete! Here are the results:")
    print("=" * 60)

    for i, candidate in enumerate(results):
        print(f"\n--- Result for Prompt #{i+1}: '{prompts[i]}' ---")
        if candidate:
            print(f"  Solution: {candidate.solution.strip()}")
            print(f"  Score: {candidate.score:.2f}")
        else:
            print("  This workflow failed to produce a valid solution.")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
