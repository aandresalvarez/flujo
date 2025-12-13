"""
Demonstrates using a Typed Pipeline Context to share state across steps.

A Typed Context is a Pydantic model that acts as shared, type-safe state for
a single pipeline run. It's perfect for passing data between non-adjacent
steps or for accumulating information throughout a workflow.
For more details, see docs/pipeline_context.md.
"""

import asyncio
from typing import Optional, cast, Type, Any

from flujo import Flujo, Step
from flujo.models import PipelineResult
from flujo.domain.agent_protocol import ContextAwareAgentProtocol
from flujo.models import BaseModel as FlujoBaseModel


# 1. Define the context model. This is our shared data structure for one run.
class ResearchContext(FlujoBaseModel):
    research_topic: str = "Unknown"
    sources_found: int = 0
    summary: Optional[str] = None


# 2. Define agents that interact with the context using the ContextAware protocol.


class PlanResearchAgent(ContextAwareAgentProtocol[str, str, ResearchContext]):
    async def run(
        self,
        data: str,
        *,
        context: ResearchContext,
        **kwargs: Any,
    ) -> str:
        """Identify the core topic and store it in the context."""
        print("🧠 Planning Agent: Analyzing task to find the core research topic.")
        topic = "The History of the Python Programming Language"
        context.research_topic = topic
        print(f"   -> Set `research_topic` in context to: '{topic}'")
        return f"Research plan for {topic}"


class GatherSourcesAgent(ContextAwareAgentProtocol[str, list[str], ResearchContext]):
    async def run(
        self,
        data: str,
        *,
        context: ResearchContext,
        **kwargs: Any,
    ) -> list[str]:
        """ "Find" sources and update a counter in the context."""
        print("📚 Gathering Sources Agent: Finding relevant articles.")
        sources = ["python.org", "Wikipedia", "A History of Computing book"]
        context.sources_found = len(sources)
        print(f"   -> Found {len(sources)} sources. Updated `sources_found` in context.")
        return sources


class SummarizeAgent(ContextAwareAgentProtocol[list[str], str, ResearchContext]):
    async def run(
        self,
        data: list[str],
        *,
        context: ResearchContext,
        **kwargs: Any,
    ) -> str:
        """Write a summary using data stored in the context."""
        print("✍️ Summarization Agent: Writing summary.")
        topic = context.research_topic
        num_sources = context.sources_found
        summary = (
            f"The summary for '{topic}' based on {num_sources} sources is complete."
            "Python was created by Guido van Rossum in the late 1980s."
        )
        context.summary = summary
        print(f"   -> Wrote summary for '{topic}' and saved to context.")
        return summary


# 3. Define the pipeline using our context-aware agents
pipeline = (
    Step.solution(PlanResearchAgent())
    >> Step.solution(GatherSourcesAgent())
    >> Step.solution(SummarizeAgent())
)

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
