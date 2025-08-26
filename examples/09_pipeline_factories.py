"""Example: Using Pipeline Factories for Transparent, Composable Workflows

This example demonstrates the new pipeline factory functions that replace
the class-based recipe approach. The factories return standard Pipeline
objects that can be:

1. Inspected and modified
2. Composed with other pipelines
3. Serialized to YAML (future)
4. Patched by AI agents (future)

This aligns with the roadmap's vision of "Workflows That Learn".
"""

import asyncio

from flujo.recipes.factories import (
    make_default_pipeline,
    make_agentic_loop_pipeline,
    run_default_pipeline,
    run_agentic_loop_pipeline,
)
from flujo.domain.models import Task, Checklist, ChecklistItem
from flujo.domain.dsl.step import Step
from flujo.domain.commands import FinishCommand
from flujo.testing.utils import StubAgent
from flujo.domain.dsl.pipeline import Pipeline


async def demonstrate_default_pipeline_factory():
    """Demonstrate the make_default_pipeline factory function."""
    print("=== Default Pipeline Factory Example ===\n")

    # Create agents for the pipeline
    review_agent = StubAgent(
        [
            Checklist(
                items=[
                    ChecklistItem(description="Check if the solution is correct"),
                    ChecklistItem(description="Verify the approach is efficient"),
                    ChecklistItem(description="Ensure the code is readable"),
                ]
            )
        ]
    )

    solution_agent = StubAgent(
        [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ]
    )

    validator_agent = StubAgent(
        [
            Checklist(
                items=[
                    ChecklistItem(description="Check if the solution is correct", completed=True),
                    ChecklistItem(description="Verify the approach is efficient", completed=False),
                    ChecklistItem(description="Ensure the code is readable", completed=True),
                ]
            )
        ]
    )

    # Create the pipeline using the factory
    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
        max_retries=2,
    )

    print("Pipeline created successfully!")
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Number of steps: {len(pipeline.steps)}")
    print(f"Step names: {[step.name for step in pipeline.steps]}")
    print()

    # Run the pipeline
    task = Task(prompt="Write a function to calculate the nth Fibonacci number")
    result = await run_default_pipeline(pipeline, task)

    print("Pipeline execution result:")
    print(f"Solution: {result.solution}")
    print(f"Score: {result.score}")
    print(f"Checklist items: {len(result.checklist.items)}")
    print()


async def demonstrate_pipeline_composability():
    """Demonstrate how pipeline factories enable composability."""
    print("=== Pipeline Composability Example ===\n")

    # Create a simple preprocessing pipeline
    preprocess_agent = StubAgent(
        ["Preprocessed: Write a function to calculate the nth Fibonacci number"]
    )

    async def preprocess_step(data: str) -> str:
        return await preprocess_agent.run(data)

    preprocess_step_s = Step.from_callable(preprocess_step, name="preprocess")
    preprocess_pipeline = Pipeline.from_step(preprocess_step_s)

    # Create the main pipeline using the factory
    review_agent = StubAgent([Checklist(items=[ChecklistItem(description="Check correctness")])])
    solution_agent = StubAgent(["def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"])
    validator_agent = StubAgent(
        [Checklist(items=[ChecklistItem(description="Check correctness", completed=True)])]
    )

    main_pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    # Compose the pipelines
    composed_pipeline = preprocess_pipeline >> main_pipeline

    print("Composed pipeline created!")
    print(f"Total steps: {len(composed_pipeline.steps)}")
    print(f"Step names: {[step.name for step in composed_pipeline.steps]}")
    print()

    # Run the composed pipeline
    task = Task(prompt="Write a function to calculate the nth Fibonacci number")
    result = await run_default_pipeline(composed_pipeline, task)

    print("Composed pipeline result:")
    print(f"Score: {result.score}")
    print()


async def demonstrate_agentic_loop_factory():
    """Demonstrate the make_agentic_loop_pipeline factory function."""
    print("=== Agentic Loop Pipeline Factory Example ===\n")

    # Create a planner that immediately finishes
    planner_agent = StubAgent([FinishCommand(final_answer="Task completed successfully!")])

    # Create an empty agent registry (no sub-agents needed for this simple example)
    agent_registry = {}

    # Create the pipeline using the factory
    pipeline = make_agentic_loop_pipeline(
        planner_agent=planner_agent,
        agent_registry=agent_registry,
        max_loops=5,
        max_retries=2,
    )

    print("Agentic loop pipeline created successfully!")
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Number of steps: {len(pipeline.steps)}")
    print(f"Step names: {[step.name for step in pipeline.steps]}")
    print()

    # Run the pipeline
    result = await run_agentic_loop_pipeline(pipeline, "Complete a simple task")

    print("Agentic loop execution result:")
    print(f"Result: {result}")
    print()


async def demonstrate_pipeline_inspection():
    """Demonstrate how pipeline factories enable inspection and modification."""
    print("=== Pipeline Inspection Example ===\n")

    # Create a pipeline
    review_agent = StubAgent([Checklist(items=[ChecklistItem(description="Check")])])
    solution_agent = StubAgent(["solution"])
    validator_agent = StubAgent(
        [Checklist(items=[ChecklistItem(description="Check", completed=True)])]
    )

    pipeline = make_default_pipeline(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent,
    )

    print("Pipeline inspection capabilities:")
    print(f"Pipeline is inspectable: {hasattr(pipeline, 'steps')}")
    print(f"Steps can be accessed: {len(pipeline.steps)}")
    print(
        f"Step configurations are visible: {[step.config.max_retries for step in pipeline.steps]}"
    )
    print()

    # Show how the pipeline structure is transparent
    print("Pipeline structure:")
    for i, step in enumerate(pipeline.steps):
        print(f"  Step {i + 1}: {step.name} (max_retries: {step.config.max_retries})")
    print()


async def main():
    """Run all demonstrations."""
    await demonstrate_default_pipeline_factory()
    await demonstrate_pipeline_composability()
    await demonstrate_agentic_loop_factory()
    await demonstrate_pipeline_inspection()

    print("=== Summary ===")
    print("✅ Pipeline factories return transparent, inspectable Pipeline objects")
    print("✅ Pipelines can be composed with other pipelines and steps")
    print("✅ Pipeline structure is visible and modifiable")
    print("✅ This enables future YAML serialization and AI-driven modifications")
    print("✅ Aligns with the roadmap's 'Workflows That Learn' vision")


if __name__ == "__main__":
    asyncio.run(main())
