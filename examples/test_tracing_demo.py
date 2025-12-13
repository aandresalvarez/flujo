#!/usr/bin/env python3
"""
Demo script to test the rich tracing functionality.

This script creates a simple pipeline to demonstrate the hierarchical trace visualization.
"""

import asyncio
from flujo.domain.dsl import Pipeline, Step
from flujo.application.runner import Flujo
from flujo.domain.models import PipelineContext


async def simple_step(input_data: str, context: PipelineContext) -> str:
    """A simple step that processes input."""
    result = f"processed_{input_data}"
    context.import_artifacts["last_result"] = result
    return result


async def another_step(input_data: str, context: PipelineContext) -> str:
    """Another simple step."""
    result = f"enhanced_{input_data}"
    context.import_artifacts["final_result"] = result
    return result


# Create a simple pipeline
pipeline = Pipeline(
    steps=[
        Step.from_callable(simple_step, name="step1"),
        Step.from_callable(another_step, name="step2"),
    ],
)


async def main():
    """Run the demo pipeline and show trace results."""
    # Create Flujo runner with tracing enabled
    flujo = Flujo(
        pipeline=pipeline,
        enable_tracing=True,
    )

    # Run the pipeline
    print("🚀 Running demo pipeline with tracing...")
    async for result in flujo.run_async("test_input"):
        # Get the final result
        pass

    print("\n✅ Pipeline completed!")

    # Get the final output from the last step result
    if result.step_history:
        last_step = result.step_history[-1]
        print(f"Final output: {last_step.output}")
        print(f"Total steps executed: {len(result.step_history)}")
    else:
        print("No step history available")

    if result.final_pipeline_context:
        print(f"Context import_artifacts: {result.final_pipeline_context.import_artifacts}")

    # Show the trace tree
    if result.trace_tree:
        print(f"\n🌳 Trace tree generated with {len(result.trace_tree.children)} root children")
        print("Use 'flujo lens trace <run_id>' to view the trace visualization")

        # Show some basic trace info
        print(f"Root span name: {result.trace_tree.name}")
        print(f"Root span status: {result.trace_tree.status}")
        if result.trace_tree.children:
            print(f"First child span: {result.trace_tree.children[0].name}")
    else:
        print("\n❌ No trace tree generated")


if __name__ == "__main__":
    asyncio.run(main())
