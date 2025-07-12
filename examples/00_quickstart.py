"""
A "Hello, World!" example demonstrating the AgenticLoop recipe.

This is the recommended starting point for building powerful, dynamic AI agents
that can make decisions and use tools to accomplish goals.
"""

import asyncio

from flujo import make_agent_async, init_telemetry
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.domain.commands import AgentCommand
from pydantic import TypeAdapter


# It's good practice to initialize telemetry at the start of your application.
init_telemetry()

# --- 1. Define the Agents (The "Team") ---


# This is our "tool" agent. It's a specialist that only knows how to search.
# In a real app, this would call a search API. We'll simulate it.
async def search_agent(query: str) -> str:
    """A simple tool agent that returns information."""
    print(f"   -> Tool Agent searching for '{query}'...")
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language."
    return "No information found."


# This is our planner agent. It decides what to do next.
PLANNER_PROMPT = """
You are a research assistant. Use the `search_agent` tool to gather facts.
When you know the answer, issue a `FinishCommand` with the final result.
"""
planner = make_agent_async(
    "openai:gpt-4o",
    PLANNER_PROMPT,
    TypeAdapter(AgentCommand),
)

# --- 2. Assemble and Run the AgenticLoop ---

print("🤖 Assembling the AgenticLoop...")

# Create the pipeline using the factory
pipeline = make_agentic_loop_pipeline(
    planner_agent=planner,
    agent_registry={"search_agent": search_agent}
)

async def main():
    # Run the pipeline
    pipeline_result = await run_agentic_loop_pipeline(pipeline, "What is Python?")
    print(f"Final result: {pipeline_result}")

    # --- 3. Inspect the Results ---
    if pipeline_result and pipeline_result.final_pipeline_context:
        print("\n✅ Loop finished!")
        final_context = pipeline_result.final_pipeline_context
        print("\n--- Agent Transcript ---")
        for log_entry in final_context.command_log:
            print(log_entry)

if __name__ == "__main__":
    asyncio.run(main())
