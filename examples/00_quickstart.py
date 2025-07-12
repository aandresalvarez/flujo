"""
A "Hello, World!" example demonstrating the AgenticLoop recipe.

This is the recommended starting point for building powerful, dynamic AI agents
that can make decisions and use tools to accomplish goals.
"""

import asyncio

from flujo import make_agent_async, init_telemetry
from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.domain.commands import AgentCommand, RunAgentCommand, AskHumanCommand, FinishCommand
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

def format_command_log(log_entry):
    """Format a command log entry for detailed output."""
    turn = log_entry.turn
    command = log_entry.generated_command
    result = log_entry.execution_result

    # Format based on command type
    if isinstance(command, RunAgentCommand):
        return f"Turn {turn}: RunAgentCommand(agent='{command.agent_name}', input='{command.input_data}') → {result}"
    elif isinstance(command, AskHumanCommand):
        return f"Turn {turn}: AskHumanCommand(question='{command.question}') → {result}"
    elif isinstance(command, FinishCommand):
        return f"Turn {turn}: FinishCommand(final_answer='{command.final_answer}') → {result}"
    else:
        return f"Turn {turn}: {type(command).__name__} → {result}"

async def main():
    # Run the pipeline
    pipeline_result = await run_agentic_loop_pipeline(pipeline, "What is Python?")
    print(f"Final result: {pipeline_result}")

    # --- 3. Inspect the Results ---
    # Only try to access final_pipeline_context if present
    if hasattr(pipeline_result, 'final_pipeline_context') and pipeline_result.final_pipeline_context:
        print("\n✅ Loop finished!")
        final_context = pipeline_result.final_pipeline_context
        print("\n--- Agent Transcript ---")
        for log_entry in final_context.command_log:
            print(format_command_log(log_entry))
    elif hasattr(pipeline_result, 'command_log') and pipeline_result.command_log:
        print("\n✅ Loop finished!")
        print("\n--- Agent Transcript ---")
        for log_entry in pipeline_result.command_log:
            print(format_command_log(log_entry))
    else:
        print("\n❌ Pipeline finished with no context log available")
        if pipeline_result:
            print(f"Pipeline result: {pipeline_result}")
        else:
            print("No pipeline result returned")

if __name__ == "__main__":
    asyncio.run(main())
