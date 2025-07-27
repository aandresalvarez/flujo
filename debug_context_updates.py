#!/usr/bin/env python3
"""
Debug script to test context updates in loop steps.
This will help us understand why the context updates aren't being applied correctly.
"""

import asyncio
from typing import Optional
from flujo import Flujo, Step, Pipeline, step
from flujo.domain.models import PipelineContext
from flujo.infra.agents import make_agent_async
from flujo.infra.settings import settings as flujo_settings

# Define a simple context for testing
class TestContext(PipelineContext):
    counter: int = 0
    is_clear: bool = False
    current_value: str = ""

# Create a simple agent for testing
TestAgent = make_agent_async(
    model=flujo_settings.default_solution_model,
    system_prompt="You are a test agent. If the input contains 'clear', respond with '[CLEAR]'. Otherwise, ask a question.",
    output_type=str,
)

@step(updates_context=True)
async def test_context_update_step(data: str, *, context: TestContext) -> dict:
    """A simple step that updates context based on agent response."""
    print(f"  [DEBUG] Step input: {data}")
    print(f"  [DEBUG] Context before: counter={context.counter}, is_clear={context.is_clear}")

    # Call the agent
    agent_response = await TestAgent.run(data, context=context)
    agent_output = getattr(agent_response, "output", agent_response)

    print(f"  [DEBUG] Agent response: {agent_output}")

    # Update context
    context.counter += 1

    if "[CLEAR]" in agent_output:
        context.is_clear = True
        return {
            "counter": context.counter,
            "is_clear": True,
            "current_value": data
        }
    else:
        context.is_clear = False
        return {
            "counter": context.counter,
            "is_clear": False,
            "current_value": f"{data} (clarified)"
        }

def exit_condition(output: dict, context: TestContext) -> bool:
    """Exit condition that checks context.is_clear."""
    print(f"  [DEBUG] Exit condition check: context.is_clear = {context.is_clear}")
    return context.is_clear

def input_mapper(initial_input: str, context: TestContext) -> str:
    """Map initial input to loop body."""
    print(f"  [DEBUG] Input mapper: {initial_input} -> {context.current_value}")
    return context.current_value or initial_input

def iteration_mapper(last_output: dict, context: TestContext, i: int) -> str:
    """Map iteration output to next input."""
    print(f"  [DEBUG] Iteration mapper: {last_output} -> {context.current_value}")
    return context.current_value

async def test_context_updates():
    """Test context updates in loop steps."""
    print("🧪 Testing Context Updates in Loop Steps")
    print("=" * 50)

    # Create the loop
    loop_body = Pipeline.from_step(test_context_update_step)

    loop_step = Step.loop_until(
        name="test_loop",
        loop_body_pipeline=loop_body,
        exit_condition_callable=exit_condition,
        max_loops=3,
        initial_input_to_loop_body_mapper=input_mapper,
        iteration_input_mapper=iteration_mapper,
    )

    # Create runner
    runner = Flujo(
        loop_step,
        pipeline_name="debug_context_updates",
        context_model=TestContext
    )

    # Test with different inputs
    test_cases = [
        "clear definition",  # Should exit after 1 iteration
        "unclear definition",  # Should continue for max iterations
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test Case {i}: {test_input} {'='*20}")

        initial_context_data = {
            "initial_prompt": test_input,
            "counter": 0,
            "is_clear": False,
            "current_value": test_input,
        }

        result = None
        async for item in runner.run_async(test_input, initial_context_data=initial_context_data):
            result = item

        if result and result.final_pipeline_context:
            final_context = result.final_pipeline_context
            print(f"\n✅ Final Results:")
            print(f"  Counter: {final_context.counter}")
            print(f"  Is Clear: {final_context.is_clear}")
            print(f"  Current Value: {final_context.current_value}")
            print(f"  Success: {result.step_history[-1].success}")
            print(f"  Attempts: {result.step_history[-1].attempts}")
        else:
            print(f"❌ Test failed: No result")

if __name__ == "__main__":
    asyncio.run(test_context_updates())
