#!/usr/bin/env python3
"""
Enhanced Loop Mappers Demo (FSD-026 Implementation)

This example demonstrates the new YAML loop mapper functionality that solves
the critical gap in data transformation for agentic loops.

The problem: A LoopStep receives a raw string but its internal agent expects
a structured dictionary. Previously, there was no declarative way to map
the initial input to the format expected by the first iteration.

The solution: New optional keys in YAML:
- initial_input_mapper: Maps LoopStep input to first iteration's body input
- iteration_input_mapper: Maps previous iteration output to next iteration input
- loop_output_mapper: Maps final successful output to LoopStep output

This enables clean, declarative, and robust conversational AI workflows.
"""

import asyncio
from typing import Any, Dict, List

from flujo import Step, Pipeline
from flujo.domain.dsl.loop import LoopStep
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent


class ConversationalContext(PipelineContext):
    """Context for conversational loop workflows."""

    initial_prompt: str = ""
    conversation_history: List[str] = []
    command_log: List[str] = []


def map_initial_input(initial_goal: str, context: ConversationalContext) -> Dict[str, Any]:
    """
    Transform the initial raw string goal into the structured input for the loop's first iteration.

    This solves the core problem: the LoopStep receives a string but the planner agent
    expects a structured dictionary.
    """
    context.initial_prompt = initial_goal
    context.command_log.append(f"Initial Goal: {initial_goal}")
    return {"initial_goal": initial_goal, "conversation_history": [], "current_step": "planning"}


def map_iteration_input(
    output: Any, context: ConversationalContext, iteration: int
) -> Dict[str, Any]:
    """
    Map the output of iteration n to the input of iteration n+1.

    This maintains conversation state and provides context for the next iteration.
    """
    if isinstance(output, str):
        context.conversation_history.append(output)
    else:
        context.conversation_history.append(str(output))

    context.command_log.append(f"Iteration {iteration}: {output}")

    return {
        "initial_goal": context.initial_prompt,
        "conversation_history": context.conversation_history,
        "current_step": "execution",
        "iteration": iteration,
    }


def is_finish_command(output: Any, context: ConversationalContext) -> bool:
    """
    Determine if the conversation should finish.

    Exit conditions:
    - User says "finish" or similar
    - Maximum iterations reached
    - Goal appears to be achieved
    """
    output_str = str(output).lower()

    # Check for explicit finish commands
    if any(word in output_str for word in ["finish", "done", "complete", "stop"]):
        context.command_log.append("Exit: User requested finish")
        return True

    # Check for goal achievement indicators
    if any(word in output_str for word in ["achieved", "completed", "successful", "ready"]):
        context.command_log.append("Exit: Goal appears achieved")
        return True

    # Check iteration limit
    if len(context.conversation_history) >= 5:
        context.command_log.append("Exit: Maximum iterations reached")
        return True

    return False


def map_loop_output(output: Any, context: ConversationalContext) -> Dict[str, Any]:
    """
    Map the final successful output to the LoopStep's output.

    This provides a clean, structured result that summarizes the entire conversation.
    """
    return {
        "final_result": output,
        "conversation_summary": context.conversation_history,
        "total_iterations": len(context.conversation_history),
        "initial_goal": context.initial_prompt,
        "command_log": context.command_log,
        "success": True,
    }


def create_conversational_loop_pipeline() -> Pipeline:
    """Create a pipeline that demonstrates the enhanced loop functionality."""

    # Create the loop body pipeline
    planner_step = Step(
        name="planner",
        agent=StubAgent(
            [
                "I'll help you build a website. Let me start by gathering requirements.",
                "Based on your goal, I recommend using a modern framework like React.",
                "Let me create a project structure and basic files for you.",
                "I'll set up the development environment and dependencies.",
                "Your website project is now ready! I've created all the necessary files and structure.",
            ]
        ),
    )

    executor_step = Step(
        name="executor",
        agent=StubAgent(
            [
                "Creating project directory...",
                "Setting up package.json...",
                "Installing dependencies...",
                "Creating component files...",
                "Project setup complete!",
            ]
        ),
    )

    loop_body = Pipeline(steps=[planner_step, executor_step])

    # Create the enhanced loop step
    loop_step = LoopStep(
        name="conversational_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=is_finish_command,
        max_loops=5,
        initial_input_to_loop_body_mapper=map_initial_input,
        iteration_input_mapper=map_iteration_input,
        loop_output_mapper=map_loop_output,
    )

    # Create the final summarization step
    summary_step = Step(
        name="generate_specification",
        agent=StubAgent(
            [
                "Based on our conversation, here's your project specification:\n\n"
                "Project: Website Development\n"
                "Framework: React\n"
                "Features: Modern UI, Responsive Design\n"
                "Status: Ready for development"
            ]
        ),
    )

    # Compose the complete pipeline
    return Pipeline(steps=[loop_step, summary_step])


def create_yaml_equivalent() -> str:
    """Create the YAML equivalent of the conversational loop pipeline."""
    return """
version: "0.1"
steps:
  - kind: loop
    name: conversational_loop
    loop:
      body:
        - name: planner
          agent:
            id: "conversation_planner"
        - name: executor
          agent:
            id: "command_executor"
      initial_input_mapper: "examples.12_enhanced_loop_mappers_demo:map_initial_input"
      iteration_input_mapper: "examples.12_enhanced_loop_mappers_demo:map_iteration_input"
      exit_condition: "examples.12_enhanced_loop_mappers_demo:is_finish_command"
      loop_output_mapper: "examples.12_enhanced_loop_mappers_demo:map_loop_output"
      max_loops: 5
  
  - kind: step
    name: generate_specification
    agent:
      id: "project_summarizer"
"""


async def run_conversational_demo() -> None:
    """Run the conversational loop demonstration."""
    print("🚀 Enhanced Loop Mappers Demo (FSD-026 Implementation)")
    print("=" * 60)
    print()

    # Create the pipeline
    pipeline = create_conversational_loop_pipeline()

    # Create context
    context = ConversationalContext(initial_prompt="build a website")

    print("📋 Pipeline Configuration:")
    print(f"  - Loop step: {pipeline.steps[0].name}")
    print(
        f"  - Has initial mapper: {pipeline.steps[0].initial_input_to_loop_body_mapper is not None}"
    )
    print(f"  - Has iteration mapper: {pipeline.steps[0].iteration_input_mapper is not None}")
    print(f"  - Has output mapper: {pipeline.steps[0].loop_output_mapper is not None}")
    print()

    print("🎯 Initial Goal:", context.initial_prompt)
    print()

    # Execute the pipeline
    print("🔄 Executing conversational loop...")
    print("-" * 40)

    # Simulate the loop execution manually to show the mapper flow
    loop_step = pipeline.steps[0]

    # Initial input mapping
    initial_input = loop_step.initial_input_to_loop_body_mapper(context.initial_prompt, context)
    print(f"📥 Initial input mapped: {initial_input}")

    # Simulate iterations
    for iteration in range(1, 4):  # Simulate 3 iterations
        print(f"\n🔄 Iteration {iteration}:")

        # Simulate loop body execution
        body_output = f"Step {iteration} completed"
        print(f"  Body output: {body_output}")

        # Check exit condition
        should_exit = loop_step.exit_condition_callable(body_output, context)
        if should_exit:
            print(f"  Exit condition met: {body_output}")
            break

        # Map to next iteration input
        next_input = loop_step.iteration_input_mapper(body_output, context, iteration)
        print(f"  Next iteration input: {next_input}")

    # Final output mapping
    final_output = loop_step.loop_output_mapper("Project setup complete", context)
    print(f"\n📤 Final output mapped: {final_output}")

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\nKey Benefits of Enhanced Loop Mappers:")
    print("1. 🎯 Clean data transformation from raw input to structured format")
    print("2. 🔄 Consistent state management across iterations")
    print("3. 📊 Rich output with conversation history and metadata")
    print("4. 🚀 Declarative YAML configuration without workarounds")
    print("5. 🔧 Backward compatible with existing loop steps")


def show_yaml_usage() -> None:
    """Show how to use the new YAML functionality."""
    print("\n📝 YAML Usage Example:")
    print("=" * 40)
    print(create_yaml_equivalent())


if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_conversational_demo())

    # Show YAML usage
    show_yaml_usage()
