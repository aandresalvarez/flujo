#!/usr/bin/env python3
"""
Quick test script to generate and save Mermaid diagrams.
"""

from flujo import Step, Pipeline
from flujo.testing.utils import StubAgent


class TestAgent:
    """Simple test agent."""
    async def run(self, data: str, **kwargs) -> str:
        return f"processed: {data}"


def create_test_pipeline():
    """Create a test pipeline with various step types."""
    
    # Simple steps
    step1 = Step("Extract", TestAgent())
    step2 = Step("Transform", TestAgent())
    step3 = Step("Load", TestAgent())
    
    # Loop step
    loop_body = Pipeline.from_step(Step("Refine", TestAgent()))
    loop_step = Step.loop_until(
        name="RefinementLoop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=3,
    )
    
    # Conditional step
    branch1 = Pipeline.from_step(Step("CodeGen", TestAgent()))
    branch2 = Pipeline.from_step(Step("QAStep", TestAgent()))
    conditional_step = Step.branch_on(
        name="Router",
        condition_callable=lambda out, ctx: "code" if "code" in str(out) else "qa",
        branches={"code": branch1, "qa": branch2},
    )
    
    # Parallel step
    parallel_step = Step.parallel(
        name="ParallelProcess",
        branches={
            "Analysis": Pipeline.from_step(Step("Analyze", TestAgent())),
            "Summary": Pipeline.from_step(Step("Summarize", TestAgent())),
        },
    )
    
    # Human step
    hitl_step = Step.human_in_the_loop("UserApproval", "Please review")
    
    # Assemble pipeline
    pipeline = step1 >> step2 >> step3 >> loop_step >> conditional_step >> parallel_step >> hitl_step
    
    return pipeline


def main():
    """Generate and save Mermaid diagram."""
    print("🔍 Generating Mermaid diagram...")
    
    pipeline = create_test_pipeline()
    mermaid_code = pipeline.to_mermaid()
    
    # Save to file
    with open("pipeline_diagram.md", "w") as f:
        f.write("# Pipeline Visualization\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")
    
    print("✅ Mermaid diagram saved to 'pipeline_diagram.md'")
    print("\n📊 You can view this diagram in:")
    print("   - GitHub (if you push this file)")
    print("   - VS Code with Mermaid extension")
    print("   - Mermaid Live Editor (https://mermaid.live)")
    print("   - Any Markdown viewer that supports Mermaid")
    
    print(f"\n📝 Mermaid code:\n{mermaid_code}")


if __name__ == "__main__":
    main() 