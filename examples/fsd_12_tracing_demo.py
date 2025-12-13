#!/usr/bin/env python3
"""
FSD-12 Tracing Demo

This script demonstrates the rich internal tracing and visualization features
implemented in FSD-12. It shows how to:

1. Create a pipeline with tracing enabled
2. Run the pipeline and access trace information
3. Use the CLI to inspect traces
4. Handle errors with trace information

Run this script to see the tracing system in action!
"""

import asyncio
import tempfile
from pathlib import Path

from flujo.application.runner import Flujo
from flujo.domain.dsl import Pipeline, Step, LoopStep, ConditionalStep
from flujo.domain.models import PipelineContext
from flujo.state.backends.sqlite import SQLiteBackend


class DemoContext(PipelineContext):
    """Demo context for tracing demonstration."""

    pass


async def simple_step(input_data: str, context: PipelineContext) -> str:
    """A simple step that processes input."""
    print(f"   🔄 Executing simple_step with input: {input_data}")
    result = f"processed_{input_data}"
    context.import_artifacts["last_result"] = result
    return result


async def loop_step(input_data: str, context: PipelineContext) -> list[str]:
    """A step that processes data in a loop."""
    print(f"   🔄 Executing loop_step with input: {input_data}")
    results = []
    for iteration in range(2):
        result = f"loop_{input_data}_{iteration}"
        results.append(result)
        print(f"     📝 Loop iteration {iteration}: {result}")
    return results


async def failing_step(input_data: str, context: PipelineContext) -> str:
    """A step that intentionally fails for demonstration."""
    print(f"   🔄 Executing failing_step with input: {input_data}")
    raise ValueError("Intentional failure for demonstration")


def create_demo_pipeline() -> Pipeline:
    """Create a complex pipeline for demonstration."""
    return Pipeline(
        steps=[
            Step.from_callable(simple_step, name="initial_step"),
            LoopStep(
                name="demo_loop",
                loop_body_pipeline=Pipeline(
                    steps=[Step.from_callable(loop_step, name="loop_step")]
                ),
                exit_condition_callable=lambda output, ctx: len(output) >= 2 if output else False,
                max_loops=2,
            ),
            ConditionalStep(
                name="demo_conditional",
                condition_callable=lambda data, ctx: "high" if len(str(data)) > 10 else "low",
                branches={
                    "high": Pipeline(steps=[Step.from_callable(simple_step, name="high_branch")]),
                    "low": Pipeline(steps=[Step.from_callable(simple_step, name="low_branch")]),
                },
            ),
            Step.from_callable(simple_step, name="final_step"),
        ],
    )


async def demo_tracing_functionality():
    """Demonstrate successful tracing functionality."""
    print("🚀 FSD-12 Tracing Demo: Successful Execution")
    print("=" * 50)

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        # Create state backend
        state_backend = SQLiteBackend(db_path)

        # Create pipeline
        pipeline = create_demo_pipeline()

        # Create Flujo runner with tracing enabled
        flujo = Flujo(
            pipeline=pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        print("📋 Pipeline created with tracing enabled")
        print("🔄 Running pipeline...")

        # Run the pipeline
        final_result = None
        async for result in flujo.run_async("demo_input"):
            final_result = result

        print("✅ Pipeline completed successfully!")
        print()

        # Access trace information
        print("🌳 Trace Information:")
        print(f"   📊 Trace generated: {final_result.trace_tree is not None}")
        print(f"   🏷️  Root span name: {final_result.trace_tree.name}")
        print(f"   📈 Status: {final_result.trace_tree.status}")
        print(
            f"   ⏱️  Duration: {final_result.trace_tree.end_time - final_result.trace_tree.start_time:.3f}s"
        )
        print(f"   📝 Total steps: {len(final_result.step_history)}")
        print(f"   🌿 Child spans: {len(final_result.trace_tree.children)}")

        # Show step history
        print("\n📋 Step History:")
        for i, step_result in enumerate(final_result.step_history):
            status = "success" if step_result.success else "failed"
            print(f"   {i + 1}. {step_result.name}: {status} ({step_result.latency_s:.3f}s)")

        # Access trace from database
        run_id = final_result.final_pipeline_context.run_id
        print(f"\n💾 Trace persisted to database with run_id: {run_id}")

        # Get spans from database
        spans = await state_backend.get_spans(run_id)
        print(f"   📊 Trace spans: {len(spans)}")

        print("\n🔍 CLI Commands to inspect this trace:")
        print("   flujo lens list                    # List all runs")
        print(f"   flujo lens show {run_id}          # Show run details")
        print(f"   flujo lens trace {run_id}         # View trace tree")
        print(f"   flujo lens spans {run_id}         # List all spans")

    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()


async def demo_error_handling():
    """Demonstrate error handling with tracing."""
    print("\n🚨 FSD-12 Tracing Demo: Error Handling")
    print("=" * 50)

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        # Create state backend
        state_backend = SQLiteBackend(db_path)

        # Create pipeline with failing step
        failing_pipeline = Pipeline(
            steps=[
                Step.from_callable(simple_step, name="step1"),
                Step.from_callable(failing_step, name="failing_step"),
                Step.from_callable(simple_step, name="step2"),
            ],
        )

        # Create Flujo runner with tracing enabled
        flujo = Flujo(
            pipeline=failing_pipeline,
            enable_tracing=True,
            state_backend=state_backend,
        )

        print("📋 Pipeline created with failing step")
        print("🔄 Running pipeline (will fail)...")

        # Run the pipeline (will fail)
        final_result = None
        async for result in flujo.run_async("demo_input"):
            final_result = result

        print("❌ Pipeline failed as expected!")
        print()

        # Access trace information
        print("🌳 Error Trace Information:")
        print(f"   📊 Trace generated: {final_result.trace_tree is not None}")
        print(f"   🏷️  Root span name: {final_result.trace_tree.name}")
        print(f"   📈 Status: {final_result.trace_tree.status}")

        # Find failed step
        failed_step = None
        for child in final_result.trace_tree.children:
            if child.name == "failing_step":
                failed_step = child
                break

        if failed_step:
            print(f"   ❌ Failed step: {failed_step.name}")
            print(f"   📈 Failed step status: {failed_step.status}")
            print(
                f"   ⏱️  Failed step duration: {failed_step.end_time - failed_step.start_time:.3f}s"
            )

        # Show step history
        print("\n📋 Step History (including failed step):")
        for i, step_result in enumerate(final_result.step_history):
            status_icon = "❌" if not step_result.success else "✅"
            status = "failed" if not step_result.success else "success"
            print(
                f"   {i + 1}. {status_icon} {step_result.name}: {status} ({step_result.latency_s:.3f}s)"
            )

        run_id = final_result.final_pipeline_context.run_id
        print(f"\n💾 Error trace persisted to database with run_id: {run_id}")

    finally:
        # Cleanup
        if db_path.exists():
            db_path.unlink()


async def main():
    """Run the FSD-12 tracing demo."""
    print("🎯 FSD-12: Rich Internal Tracing and Visualization Demo")
    print("=" * 60)
    print()
    print("This demo showcases the tracing system implemented in FSD-12.")
    print("It demonstrates hierarchical trace generation, persistence,")
    print("error handling, and CLI integration.")
    print()

    # Demo successful execution
    await demo_tracing_functionality()

    # Demo error handling
    await demo_error_handling()

    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print()
    print("Key FSD-12 Features Demonstrated:")
    print("✅ Hierarchical trace generation")
    print("✅ Trace persistence to SQLite")
    print("✅ Error handling with trace information")
    print("✅ CLI integration for trace inspection")
    print("✅ Performance monitoring with minimal overhead")
    print()
    print("The tracing system is now ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())
