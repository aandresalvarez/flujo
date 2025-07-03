"""
Example: Pipeline Visualization and Testing Ergonomics

This example demonstrates two advanced developer tooling features:

1. Advanced Pipeline Visualization: Using the `.to_mermaid()` method to generate
   comprehensive visual representations of complex pipelines.

2. Enhanced Testing Ergonomics: Using the `override_agent` context manager to
   simplify unit testing of application code that uses flujo pipelines.

For more details, see the FSD-2024-005 specification.
"""

import asyncio
from typing import Any
from pydantic import BaseModel

from flujo import Flujo, Step, Pipeline
from flujo.testing.utils import override_agent, StubAgent


# ============================================================================
# Example 1: Advanced Pipeline Visualization
# ============================================================================

class DataProcessor:
    """Example agent for data processing."""
    
    async def run(self, data: str, **kwargs: Any) -> str:
        return f"processed: {data}"


class Validator:
    """Example validation agent."""
    
    async def run(self, data: str, **kwargs: Any) -> bool:
        return len(data) > 10


def create_complex_pipeline() -> Pipeline[str, str]:
    """Create a complex pipeline with loops, conditionals, and parallel execution."""
    
    # Simple processing steps
    extract_step = Step("Extract", DataProcessor())
    validate_step = Step("Validate", Validator(), max_retries=3)
    transform_step = Step("Transform", DataProcessor())
    
    # Loop body: refine the data
    refine_step = Step("Refine", DataProcessor())
    loop_body = Pipeline.from_step(refine_step)
    
    # Loop step
    loop_step = Step.loop_until(
        name="RefinementLoop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda out, ctx: len(str(out)) > 50,
        max_loops=5,
    )
    
    # Conditional step with branches
    code_step = Step("GenerateCode", DataProcessor())
    qa_step = Step("AnswerQuestion", DataProcessor())
    
    conditional_step = Step.branch_on(
        name="TaskRouter",
        condition_callable=lambda out, ctx: "code" if "code" in str(out) else "qa",
        branches={
            "code": Pipeline.from_step(code_step),
            "qa": Pipeline.from_step(qa_step),
        },
    )
    
    # Parallel step
    parallel_step = Step.parallel(
        name="ParallelProcess",
        branches={
            "Analysis": Pipeline.from_step(Step("Analyze", DataProcessor())),
            "Summary": Pipeline.from_step(Step("Summarize", DataProcessor())),
        },
    )
    
    # Human-in-the-loop step
    hitl_step = Step.human_in_the_loop("UserApproval", "Please review the final result")
    
    # Assemble the complex pipeline
    pipeline = (
        extract_step >>
        validate_step >>
        transform_step >>
        loop_step >>
        conditional_step >>
        parallel_step >>
        hitl_step
    )
    
    return pipeline


def demonstrate_visualization():
    """Demonstrate the pipeline visualization feature."""
    print("🔍 Demonstrating Advanced Pipeline Visualization")
    print("=" * 60)
    
    pipeline = create_complex_pipeline()
    
    # Generate the Mermaid diagram
    mermaid_diagram = pipeline.to_mermaid()
    
    print("Generated Mermaid diagram:")
    print("-" * 40)
    print(mermaid_diagram)
    print("-" * 40)
    
    print("\nKey visualization features:")
    print("✅ Different shapes for different step types:")
    print("   - Standard steps: rectangles [\"Step Name\"]")
    print("   - Loop steps: stadiums (\"Loop: Name\")")
    print("   - Conditional steps: diamonds {\"Branch: Name\"}")
    print("   - Parallel steps: hexagons {{\"Parallel: Name\"}}")
    print("   - Human steps: parallelograms [/Human: Name/]")
    
    print("\n✅ Configuration annotations:")
    print("   - Validation steps: 🛡️ icon")
    print("   - Retry steps: dashed edges (-.->)")
    
    print("\n✅ Control flow visualization:")
    print("   - Loop bodies shown in subgraphs")
    print("   - Branch pipelines in separate subgraphs")
    print("   - Parallel execution with join nodes")
    
    print("\nYou can copy this Mermaid diagram into:")
    print("- GitHub markdown files")
    print("- Mermaid Live Editor (https://mermaid.live)")
    print("- IDE plugins that support Mermaid")
    print("- Documentation tools like MkDocs")


# ============================================================================
# Example 2: Enhanced Testing Ergonomics
# ============================================================================

class ProductionAgent:
    """A production agent that might be expensive or slow to run."""
    
    async def run(self, data: str, **kwargs: Any) -> str:
        # Simulate expensive operation
        await asyncio.sleep(0.1)
        return f"expensive_result: {data.upper()}"


class ApplicationService:
    """Example application service that uses flujo pipelines internally."""
    
    def __init__(self):
        self.pipeline = (
            Step("Process", ProductionAgent()) >>
            Step("Validate", ProductionAgent())
        )
        self.runner = Flujo(self.pipeline)
    
    async def process_data(self, data: str) -> str:
        """Process data using the internal pipeline."""
        result = None
        async for item in self.runner.run_async(data):
            result = item
        return result.step_history[-1].output


async def demonstrate_testing_ergonomics():
    """Demonstrate the testing ergonomics feature."""
    print("\n\n🧪 Demonstrating Enhanced Testing Ergonomics")
    print("=" * 60)
    
    # Create the application service
    service = ApplicationService()
    
    # Create test agents
    fast_test_agent = StubAgent(["test_processed", "test_validated"])
    
    print("Testing application service with overridden agents...")
    
    # Test the service with overridden agents
    with override_agent(service.pipeline.steps[0], fast_test_agent):
        with override_agent(service.pipeline.steps[1], fast_test_agent):
            # This will run much faster and predictably
            result = await service.process_data("test_input")
            print(f"Test result: {result}")
    
    print("\nKey testing benefits:")
    print("✅ Fast execution: No expensive operations during testing")
    print("✅ Predictable outputs: StubAgent provides controlled responses")
    print("✅ Automatic cleanup: Original agents are restored automatically")
    print("✅ Exception safety: Agents restored even if tests fail")
    print("✅ Simple syntax: Clean context manager interface")


async def demonstrate_complex_testing_scenario():
    """Demonstrate a more complex testing scenario."""
    print("\n\n🔬 Complex Testing Scenario")
    print("=" * 40)
    
    # Create a complex pipeline for testing
    complex_pipeline = (
        Step("Step1", ProductionAgent()) >>
        Step("Step2", ProductionAgent()) >>
        Step("Step3", ProductionAgent())
    )
    
    # Create different test agents for different scenarios
    success_agent = StubAgent(["success_output"])
    failure_agent = StubAgent([RuntimeError("Test failure")])
    slow_agent = StubAgent(["slow_output"])
    
    print("Testing different scenarios:")
    
    # Scenario 1: Success case
    print("\n1. Testing success scenario...")
    with override_agent(complex_pipeline.steps[0], success_agent):
        with override_agent(complex_pipeline.steps[1], success_agent):
            with override_agent(complex_pipeline.steps[2], success_agent):
                try:
                    result = await complex_pipeline.steps[0].arun("test")
                    print(f"   Success: {result}")
                except Exception as e:
                    print(f"   Error: {e}")
    
    # Scenario 2: Failure case
    print("\n2. Testing failure scenario...")
    with override_agent(complex_pipeline.steps[1], failure_agent):
        try:
            result = await complex_pipeline.steps[1].arun("test")
            print(f"   Success: {result}")
        except Exception as e:
            print(f"   Expected error: {e}")
    
    print("\n✅ All test scenarios completed successfully!")
    print("✅ Original agents were automatically restored")


async def main():
    """Run the demonstration."""
    print("🚀 Pipeline Visualization and Testing Ergonomics Demo")
    print("=" * 70)
    
    # Demonstrate visualization
    demonstrate_visualization()
    
    # Demonstrate testing ergonomics
    await demonstrate_testing_ergonomics()
    
    # Demonstrate complex testing
    await demonstrate_complex_testing_scenario()
    
    print("\n\n🎉 Demo completed successfully!")
    print("\nThese features help developers:")
    print("📊 Understand complex pipeline structures at a glance")
    print("🧪 Write faster, more reliable unit tests")
    print("🔧 Debug pipeline issues more effectively")
    print("📚 Create better documentation with visual diagrams")


if __name__ == "__main__":
    asyncio.run(main()) 