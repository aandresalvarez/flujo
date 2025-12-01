#!/usr/bin/env python3
"""
Enhanced Loop Mappers Demo - Current Flujo 0.4.37 Compatible Version

This demo shows:
1. How to work with current Flujo loop limitations
2. What enhanced loop mappers would enable (FSD-026)
3. The difference between current workarounds and future declarative approach

Note: The enhanced loop mapper functionality shown in the "Future" section
is not yet available in Flujo 0.4.37. This demonstrates the concept and
shows how it would simplify conversational AI workflows.
"""

from typing import Any, List
from flujo.domain.dsl.pipeline import Pipeline
from flujo.domain.dsl.step import Step
from flujo.domain.models import PipelineContext
from flujo.testing.utils import StubAgent
from flujo.type_definitions.common import JSONObject


class ConversationalContext(PipelineContext):
    """Extended context for conversational workflows."""

    initial_prompt: str
    conversation_history: List[str] = []
    command_log: List[str] = []
    current_goal: str = ""
    clarification_count: int = 0


def map_initial_input(context: ConversationalContext) -> JSONObject:
    """
    Maps the initial LoopStep input to structured format for first iteration.

    In current Flujo, this would need to be done manually in each step.
    With enhanced loop mappers, this would be automatic.
    """
    return {
        "goal": context.initial_prompt,
        "conversation_history": [],
        "iteration": 0,
        "status": "initial",
    }


def map_iteration_input(context: ConversationalContext, previous_output: Any) -> JSONObject:
    """
    Maps previous iteration output to next iteration input.

    In current Flujo, this requires manual context management.
    With enhanced loop mappers, this would be declarative.
    """
    # Add to conversation history
    if isinstance(previous_output, str):
        context.conversation_history.append(previous_output)

    context.clarification_count += 1

    return {
        "goal": context.current_goal,
        "conversation_history": context.conversation_history,
        "iteration": context.clarification_count,
        "status": "clarifying",
    }


def is_finish_command(context: ConversationalContext, output: Any) -> bool:
    """Determines if the loop should exit."""
    if isinstance(output, str):
        return output.startswith("COMPLETE:") or context.clarification_count >= 3
    return context.clarification_count >= 3


def map_loop_output(context: ConversationalContext, final_output: Any) -> JSONObject:
    """
    Maps the final successful output to LoopStep result.

    In current Flujo, this requires manual post-processing.
    With enhanced loop mappers, this would be automatic.
    """
    return {
        "final_goal": context.current_goal,
        "conversation_history": context.conversation_history,
        "total_clarifications": context.clarification_count,
        "final_output": final_output,
        "status": "completed",
    }


def create_current_flujo_workaround_pipeline() -> Pipeline:
    """
    Current Flujo 0.4.37 approach - requires manual workarounds.

    This shows what you have to do now to achieve the same functionality.
    """
    # Create agents that simulate the conversation
    planner_agent = StubAgent(
        [
            "I need to understand your goal better. What specific outcome are you looking for?",
            "Can you provide more context about the environment?",
            "COMPLETE: I now understand you want to create a data processing pipeline with error handling.",
        ]
    )

    executor_agent = StubAgent(
        [
            "I'll help you with that. Let me break this down into steps.",
            "Based on your clarification, here's the refined approach.",
            "Perfect! Here's your final implementation plan.",
        ]
    )

    # Current Flujo approach - manual step-by-step conversation
    steps = [
        # Step 1: Get initial goal
        Step(
            name="get_initial_goal", agent=planner_agent, input="What would you like to accomplish?"
        ),
        # Step 2: Manual conversation loop (simulated)
        Step(
            name="conversation_loop",
            agent=executor_agent,
            input="Let's clarify your goal step by step.",
        ),
        # Step 3: Manual context building
        Step(
            name="build_context",
            agent=StubAgent(["Context built manually"]),
            input="Building conversation context...",
        ),
    ]

    return Pipeline(steps=steps)


def create_future_enhanced_pipeline() -> Pipeline:
    """
    Future approach with enhanced loop mappers (FSD-026).

    This shows what would be possible with the enhanced functionality.
    Note: This is conceptual and won't work in current Flujo 0.4.37.
    """
    # This is what the future YAML would look like:
    future_yaml = """
    - kind: loop
      name: conversational_loop
      loop:
        body:
          - kind: step
            name: planner
            uses: agents.conversation_planner
          - kind: step
            name: executor
            uses: agents.command_executor
        initial_input_mapper: "examples.12_enhanced_loop_mappers_demo:map_initial_input"
        iteration_input_mapper: "examples.12_enhanced_loop_mappers_demo:map_iteration_input"
        exit_condition: "examples.12_enhanced_loop_mappers_demo:is_finish_command"
        loop_output_mapper: "examples.12_enhanced_loop_mappers_demo:map_loop_output"
        max_loops: 5
    """

    print("Future YAML Configuration (FSD-026):")
    print(future_yaml)

    # For now, return a simple pipeline that demonstrates the concept
    return Pipeline(
        steps=[
            Step(
                name="concept_demo",
                agent=StubAgent(["Enhanced loop mappers would enable this declarative approach"]),
                input="This is what enhanced loop mappers would enable",
            )
        ]
    )


def run_current_workaround_demo():
    """Demonstrates the current Flujo 0.4.37 approach."""
    print("=" * 60)
    print("CURRENT FLUJO 0.4.37 APPROACH")
    print("=" * 60)
    print("This shows what you have to do now to achieve conversational loops.\n")

    # Create context
    context = ConversationalContext(
        initial_prompt="I want to create a data processing pipeline",
        conversation_history=[],
        command_log=[],
        current_goal="",
        clarification_count=0,
    )

    # Simulate the manual conversation flow
    print("1. Initial input received:", context.initial_prompt)
    print("2. Manual step: Process and structure the input")
    print("3. Manual step: Create conversation context")
    print("4. Manual step: Handle each iteration manually")
    print("5. Manual step: Build conversation history")
    print("6. Manual step: Transform final output")

    print("\nCurrent Limitations:")
    print("- No automatic input mapping")
    print("- No automatic context preservation")
    print("- No automatic output transformation")
    print("- Requires manual step-by-step management")

    return context


def show_future_enhanced_approach():
    """Shows what enhanced loop mappers would enable."""
    print("\n" + "=" * 60)
    print("FUTURE ENHANCED LOOP MAPPER APPROACH (FSD-026)")
    print("=" * 60)
    print("This shows what would be possible with enhanced loop mappers.\n")

    print("Enhanced Loop Mapper Keys:")
    print("- initial_input_mapper: Automatically maps LoopStep input to first iteration")
    print("- iteration_input_mapper: Automatically maps between iterations")
    print("- loop_output_mapper: Automatically transforms final output")
    print("- exit_condition: Determines when to stop (already supported)")

    print("\nBenefits:")
    print("- Declarative YAML configuration")
    print("- Automatic data transformation")
    print("- Automatic context preservation")
    print("- Clean, maintainable workflows")
    print("- No more adapter steps or manual context management")


def show_yaml_usage():
    """Shows the YAML equivalent for both approaches."""
    print("\n" + "=" * 60)
    print("YAML COMPARISON")
    print("=" * 60)

    print("Current Flujo 0.4.37 (Manual Approach):")
    current_yaml = """
    - kind: step
      name: get_initial_goal
      agent:
        id: "flujo.builtins.ask_user"
      input: "What would you like to accomplish?"
    
    - kind: step
      name: process_goal
      uses: agents.goal_processor
      input: "{{ previous_step }}"
    
    - kind: step
      name: build_context
      uses: agents.context_builder
      input: "{{ context.processed_goal }}"
    
    - kind: loop
      name: clarification_loop
      loop:
        body:
          - kind: step
            name: ask_clarification
            agent:
              id: "flujo.builtins.ask_user"
            input: "{{ context.clarification_question }}"
          - kind: step
            name: process_response
            uses: agents.response_processor
            input: "{{ previous_step }}"
        exit_condition: "helpers:is_goal_complete"
        max_loops: 5
    """
    print(current_yaml)

    print("\nFuture Enhanced Approach (FSD-026):")
    future_yaml = """
    - kind: loop
      name: conversational_loop
      loop:
        body:
          - kind: step
            name: planner
            uses: agents.conversation_planner
          - kind: step
            name: executor
            uses: agents.command_executor
        initial_input_mapper: "skills.helpers:map_initial_goal"
        iteration_input_mapper: "skills.helpers:map_conversation_state"
        exit_condition: "skills.helpers:is_conversation_complete"
        loop_output_mapper: "skills.helpers:map_final_result"
        max_loops: 5
    """
    print(future_yaml)

    print("\nKey Differences:")
    print("1. Enhanced approach eliminates manual context management")
    print("2. Enhanced approach provides automatic data transformation")
    print("3. Enhanced approach maintains conversation state automatically")
    print("4. Enhanced approach produces rich, structured output")


def main():
    """Main demo function."""
    print("Enhanced Loop Mappers Demo - Flujo 0.4.37 Compatible")
    print("=" * 60)
    print("This demo shows the difference between current Flujo limitations")
    print("and what enhanced loop mappers would enable.\n")

    # Show current approach
    run_current_workaround_demo()

    # Show future approach
    show_future_enhanced_approach()

    # Show YAML comparison
    show_yaml_usage()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Current Flujo 0.4.37:")
    print("- Works but requires manual workarounds")
    print("- No automatic input/output mapping")
    print("- Manual context management needed")
    print("- More verbose YAML configuration")

    print("\nFuture Enhanced Approach (FSD-026):")
    print("- Declarative YAML configuration")
    print("- Automatic data transformation")
    print("- Automatic context preservation")
    print("- Clean, maintainable workflows")

    print("\nThe enhanced loop mappers would solve the exact problems")
    print("encountered in conversational AI workflows, making them")
    print("much simpler and more robust.")


if __name__ == "__main__":
    main()
