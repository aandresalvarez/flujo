#!/usr/bin/env python3
"""
Test script demonstrating different detail levels for pipeline visualization.
"""

from flujo import Step, Pipeline
from flujo.testing.utils import StubAgent


class TestAgent:
    """Simple test agent."""
    async def run(self, data: str, **kwargs) -> str:
        return f"processed: {data}"


def create_complex_pipeline():
    """Create a complex pipeline to test different detail levels."""
    
    # Simple processing steps
    extract_step = Step("Extract", TestAgent())
    validate_step = Step("Validate", TestAgent(), max_retries=3)
    transform_step = Step("Transform", TestAgent())
    
    # Loop body: refine the data
    refine_step = Step("Refine", TestAgent())
    loop_body = Pipeline.from_step(refine_step)
    
    # Loop step
    loop_step = Step.loop_until(
        name="RefinementLoop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=lambda out, ctx: True,
        max_loops=5,
    )
    
    # Conditional step with branches
    code_step = Step("GenerateCode", TestAgent())
    qa_step = Step("AnswerQuestion", TestAgent())
    
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
            "Analysis": Pipeline.from_step(Step("Analyze", TestAgent())),
            "Summary": Pipeline.from_step(Step("Summarize", TestAgent())),
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


def create_simple_pipeline():
    """Create a simple pipeline for comparison."""
    step1 = Step("Extract", TestAgent())
    step2 = Step("Transform", TestAgent())
    step3 = Step("Load", TestAgent())
    
    return step1 >> step2 >> step3


def save_mermaid_to_file(mermaid_code: str, filename: str, title: str):
    """Save Mermaid code to a markdown file."""
    with open(filename, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")


def main():
    """Test all detail levels."""
    print("🔍 Testing Pipeline Visualization Detail Levels")
    print("=" * 60)
    
    # Test simple pipeline
    print("\n📊 Simple Pipeline:")
    simple_pipeline = create_simple_pipeline()
    
    print(f"Complexity Score: {simple_pipeline._calculate_complexity_score()}")
    print(f"Auto-detected level: {simple_pipeline._determine_optimal_detail_level()}")
    
    # Test complex pipeline
    print("\n📊 Complex Pipeline:")
    complex_pipeline = create_complex_pipeline()
    
    print(f"Complexity Score: {complex_pipeline._calculate_complexity_score()}")
    print(f"Auto-detected level: {complex_pipeline._determine_optimal_detail_level()}")
    
    # Generate all detail levels for complex pipeline
    detail_levels = ["high", "medium", "low", "auto"]
    
    for level in detail_levels:
        print(f"\n🎯 Generating {level.upper()} detail level...")
        
        if level == "auto":
            mermaid_code = complex_pipeline.to_mermaid_with_detail_level("auto")
            actual_level = complex_pipeline._determine_optimal_detail_level()
            filename = f"pipeline_{level}_{actual_level}.md"
            title = f"Pipeline Visualization - {level.upper()} (Auto-detected: {actual_level.upper()})"
        else:
            mermaid_code = complex_pipeline.to_mermaid_with_detail_level(level)
            filename = f"pipeline_{level}.md"
            title = f"Pipeline Visualization - {level.upper()} Detail"
        
        save_mermaid_to_file(mermaid_code, filename, title)
        print(f"✅ Saved to {filename}")
        
        # Show a preview of the Mermaid code
        lines = mermaid_code.split('\n')
        preview_lines = lines[:10]  # Show first 10 lines
        if len(lines) > 10:
            preview_lines.append("    ...")
        preview = '\n'.join(preview_lines)
        print(f"📝 Preview:\n{preview}")
    
    # Generate PNG files for comparison
    print("\n🖼️  Generating PNG files for visual comparison...")
    import subprocess
    
    for level in detail_levels:
        if level == "auto":
            actual_level = complex_pipeline._determine_optimal_detail_level()
            filename = f"pipeline_{level}_{actual_level}.md"
            png_filename = f"pipeline_{level}_{actual_level}.png"
        else:
            filename = f"pipeline_{level}.md"
            png_filename = f"pipeline_{level}.png"
        
        try:
            subprocess.run([
                "npx", "@mermaid-js/mermaid-cli", 
                "-i", filename, 
                "-o", png_filename
            ], check=True, capture_output=True)
            print(f"✅ Generated {png_filename}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not generate {png_filename} (Mermaid CLI not available)")
    
    print("\n🎉 Detail level testing completed!")
    print("\n📁 Generated files:")
    print("   - pipeline_high.md - Full detail with all features")
    print("   - pipeline_medium.md - Simplified structure with emojis")
    print("   - pipeline_low.md - Minimal overview")
    print("   - pipeline_auto_*.md - AI-determined optimal level")
    print("\n📊 Key differences:")
    print("   HIGH: Full subgraphs, all annotations, detailed edges")
    print("   MEDIUM: Simplified shapes, emojis, no subgraphs")
    print("   LOW: Grouped steps, minimal information")
    print("   AUTO: AI chooses based on complexity score")


if __name__ == "__main__":
    main() 