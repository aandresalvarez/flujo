"""
Demonstrates using a Typed Pipeline Context to share state across steps.

A Typed Context is a Pydantic model that acts as a shared "scratchpad" for
a single pipeline run. It's perfect for passing data between non-adjacent
steps or for accumulating information throughout a workflow.
For more details, see docs/pipeline_context.md.
"""

import asyncio
from typing import Optional, cast, Type

from flujo.domain.models import BaseModel as FlujoBaseModel

from flujo import Flujo, PipelineResult, step


# 1. Define the context model. This is our shared data structure for one run.
class ResearchContext(FlujoBaseModel):
    research_topic: str = "Unknown"
    sources_found: int = 0
    summary: Optional[str] = None


# 2. Define agents that interact with the context.
#    They declare a keyword-only `pipeline_context` argument to get access.
@step
async def plan_research_agent(task: str, *, pipeline_context: ResearchContext) -> str:
    """This agent identifies the core topic and saves it to the context."""
    print("🧠 Planning Agent: Analyzing task to find the core research topic.")
    # In a real app, an LLM would extract the topic. We'll simulate it.
    topic = "The History of the Python Programming Language"
    pipeline_context.research_topic = topic
    print(f"   -> Set `research_topic` in context to: '{topic}'")
    return f"Research plan for {topic}"


@step
async def gather_sources_agent(
    plan: str, *, pipeline_context: ResearchContext
) -> list[str]:
    """This agent "finds" sources and updates a counter in the context."""
    print("📚 Gathering Sources Agent: Finding relevant articles.")
    sources = ["python.org", "Wikipedia", "A History of Computing book"]
    pipeline_context.sources_found = len(sources)
    print(f"   -> Found {len(sources)} sources. Updated `sources_found` in context.")
    return sources


@step
async def summarize_agent(
    sources: list[str], *, pipeline_context: ResearchContext
) -> str:
    """
    This agent uses data (`research_topic`, `sources_found`) from the context
    to write its summary, demonstrating how a later step can use data from earlier ones.
    """
    print("✍️ Summarization Agent: Writing summary.")
    topic = pipeline_context.research_topic  # <-- Reading from context
    num_sources = pipeline_context.sources_found
    summary = (
        f"The summary for '{topic}' based on {num_sources} sources is complete. "
        "Python was created by Guido van Rossum in the late 1980s."
    )
    pipeline_context.summary = summary
    print(f"   -> Wrote summary for '{topic}' and saved to context.")
    return summary


# 3. Define the pipeline using the step decorator
pipeline = plan_research_agent >> gather_sources_agent >> summarize_agent

# 4. Initialize the Flujo runner, telling it to use our context model.

runner = Flujo(pipeline, context_model=cast(Type[FlujoBaseModel], ResearchContext))


async def main() -> None:
    print("🚀 Starting multi-step research pipeline with a shared context...\n")
    result: PipelineResult | None = None
    async for item in runner.run_async("Create a report on Python's history."):
        result = item

    # 5. Inspect the final state of the context after the run is complete.
    print("\n✅ Pipeline finished!")
    assert result is not None
    final_context = cast(ResearchContext, result.final_pipeline_context)

    print("\nFinal Context State:")
    print(f"  - Topic: {final_context.research_topic}")
    print(f"  - Sources Found: {final_context.sources_found}")
    print(f"  - Summary: {final_context.summary}")


if __name__ == "__main__":
    asyncio.run(main())
