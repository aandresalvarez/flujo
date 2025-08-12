"""A "Hello, World!" example demonstrating the agentic loop pipeline factory.

This example shows how to create an explorative agent workflow using the
make_agentic_loop_pipeline factory function.
"""

import asyncio

from flujo.recipes.factories import make_agentic_loop_pipeline, run_agentic_loop_pipeline
from flujo.agents import make_agent_async
from flujo.infra import init_telemetry
from flujo.domain.commands import (
    AgentCommand,
    FinishCommand,
    RunAgentCommand,
    AskHumanCommand,
)
from pydantic import TypeAdapter


async def search_agent(query: str) -> str:
    """A simple tool agent that returns information."""
    if "python" in query.lower():
        return "Python is a high-level, general-purpose programming language."
    return "No information found."


async def main():
    """Run the agentic loop example."""
    # Enable telemetry (optional but recommended)
    init_telemetry()

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

    print("🤖 Assembling the agentic loop pipeline...")

    # Create the pipeline using the factory
    pipeline = make_agentic_loop_pipeline(
        planner_agent=planner, agent_registry={"search_agent": search_agent}
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

    # Run the pipeline
    print("🚀 Running the agentic loop pipeline...")
    result = await run_agentic_loop_pipeline(pipeline, "What is Python?")

    print("\n📋 Command Log:")
    for entry in result.final_pipeline_context.command_log:
        print(f"  {format_command_log(entry)}")

    print(f"\n🎯 Final Answer: {result.final_pipeline_context.command_log[-1].execution_result}")


if __name__ == "__main__":
    asyncio.run(main())
