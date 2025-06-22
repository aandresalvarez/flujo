"""
A "Hello, World!" example demonstrating the AgenticLoop recipe.

This is the recommended starting point for building powerful, dynamic AI agents
that can make decisions and use tools to accomplish goals.
"""

from typing import cast

from flujo.recipes import AgenticLoop
from flujo import make_agent_async, init_telemetry
from flujo.domain.commands import AgentCommand, FinishCommand, RunAgentCommand
from flujo.domain.models import PipelineContext
from pydantic import TypeAdapter


# It's good practice to initialize telemetry at the start of your application.
init_telemetry()

# --- 1. Define the Agents (The "Team") ---


# This is our "tool" agent. It's a specialist that only knows how to search.
# In a real app, this would call a search API. We'll simulate it.
async def search_agent(query: str) -> str:
    print(f"   -> Tool Agent: Searching for '{query}'...")
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language."
    return "No information found."


# This is our planner agent. It decides what to do next.
PLANNER_PROMPT = """
You are a research assistant. Your goal is to answer the user's question.
You can use the 'search_agent' to find information.
When you have the answer, use the FinishCommand to provide the final result.
"""
planner_agent = make_agent_async(
    model="openai:gpt-4o",
    system_prompt=PLANNER_PROMPT,
    output_type=TypeAdapter(AgentCommand),  # type: ignore[arg-type]
)

# --- 2. Assemble and Run the AgenticLoop ---

print("🤖 Assembling the AgenticLoop...")
agentic_loop = AgenticLoop(
    planner_agent=planner_agent,
    agent_registry={"search_agent": search_agent},  # type: ignore[dict-item]
)

initial_goal = "What is Python?"
print(f"🎯 Setting initial goal: '{initial_goal}'")
print("🧠 Running the loop...")

result = agentic_loop.run(initial_goal)

# --- 3. Inspect the Results ---
if result and result.final_pipeline_context:
    print("\n✅ Loop finished!")
    final_context = cast(PipelineContext, result.final_pipeline_context)
    print("\n--- Agent Transcript ---")
    for log_entry in final_context.command_log:
        command_type = log_entry.generated_command.type
        print(f"Turn #{log_entry.turn}: Planner decided to '{command_type}'")
        if isinstance(log_entry.generated_command, RunAgentCommand):
            print(
                f"   - Details: Run agent '{log_entry.generated_command.agent_name}' with input '{log_entry.generated_command.input_data}'"
            )
            print(f"   - Result: '{log_entry.execution_result}'")
        elif isinstance(log_entry.generated_command, FinishCommand):
            print(f"   - Final Answer: '{log_entry.execution_result}'")
    print("----------------------")
else:
    print("\n❌ The loop failed to produce a result.")
